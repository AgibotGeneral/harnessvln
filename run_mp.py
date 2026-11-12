import argparse
import random
import os
import gzip
import json
import re
from copy import deepcopy
import glob
from pprint import pprint
import time

from dotenv import load_dotenv

# Load local endpoint credentials before multiprocessing uses spawn. Existing
# shell variables take precedence over values in the gitignored .env file.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines import register_extensions
from vlnce_baselines.config.default import get_config
from vlnce_baselines.utils.misc import seed_everything
    
def get_episode_ids_from_config(config):
    data_path = config.TASK_CONFIG.DATASET.DATA_PATH
    split = config.TASK_CONFIG.DATASET.SPLIT
    if _is_rxr_dataset(config): role = config.TASK_CONFIG.DATASET.ROLES
    logger.info(split)
    if _is_rxr_dataset(config): data_path = data_path.format(split=split,role=role[0])
    else: data_path = data_path.format(split=split)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Episode data file not found: {data_path}")

    if data_path.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'

    with open_fn(data_path, mode) as f:
        data = json.load(f)
        episodes = data.get("episodes", [])
        # RxR ships every language in one file, but the dataset class drops all
        # but DATASET.LANGUAGES when a worker loads it (habitat_extensions/task.py
        # `_language_from_episode`). Applying the same filter here is what keeps the
        # queue and the workers talking about the same episode set: without it the
        # queue holds all 11006 rxr episodes while a worker can only ever match the
        # 3669 English ones, and every non-English id it pulls resolves to an empty
        # dataset (see `eval_dynamic`, which now skips those rather than hanging).
        if _is_rxr_dataset(config):
            languages = list(getattr(config.TASK_CONFIG.DATASET, 'LANGUAGES', []) or [])
            if languages and '*' not in languages:
                allowed = set(languages)
                before = len(episodes)
                episodes = [
                    e for e in episodes
                    if e.get("instruction", {}).get("language") in allowed
                ]
                logger.info(
                    f"RxR language filter {sorted(allowed)}: "
                    f"{before} -> {len(episodes)} episodes"
                )
        # Return detailed info: id and scene_id
        episode_info = [{"id": str(e["episode_id"]), "scene_id": e["scene_id"]} for e in episodes]
    return episode_info


def extract_episode_ids_from_log(log_path: str) -> set:
    """Extract numeric episode ids from log lines like: ep187:{...}

    Returns a set of episode id strings, e.g. {"187", "1051"}.
    """
    if not log_path:
        return set()
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    pattern = re.compile(r"\bep(\d+)\s*:")
    episode_ids = set()
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episode_ids.add(m.group(1))
    return episode_ids


def _is_rxr_dataset(cfg) -> bool:
    try:
        dp = getattr(cfg.TASK_CONFIG.DATASET, 'DATA_PATH', None)
        if dp and 'rxr' in str(dp).lower():
            return True
        name = (
            getattr(cfg.TASK_CONFIG.DATASET, 'DATASET', None)
            or getattr(cfg.TASK_CONFIG.DATASET, 'NAME', None)
            or getattr(cfg.TASK_CONFIG.DATASET, 'DATASET_NAME', None)
        )
        if name and 'rxr' in str(name).lower():
            return True
    except Exception:
        pass
    return False


def validate_model_environment() -> None:
    from vlnce_baselines.model_config import ModelConfig
    ModelConfig.from_env()


def validate_cuda_environment() -> None:
    """Fail before spawning workers when the Torch wheel cannot run this GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This evaluator requires a CUDA-enabled "
            "PyTorch installation and a visible NVIDIA GPU."
        )

    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = torch.cuda.get_arch_list()
    if required_arch not in compiled_arches:
        raise RuntimeError(
            f"GPU {torch.cuda.get_device_name(0)} requires {required_arch}, but "
            f"PyTorch {torch.__version__} only contains {compiled_arches}. "
            "Install a newer CUDA wheel before starting evaluation."
        )

    # Force one kernel launch here so driver/runtime mismatches fail in the
    # parent process instead of after every episode has loaded a Habitat scene.
    probe = torch.ones(1, device="cuda")
    probe.add_(1)
    torch.cuda.synchronize()
    logger.info(
        f"CUDA preflight passed: {torch.cuda.get_device_name(0)}, "
        f"capability={required_arch}, torch={torch.__version__}, "
        f"torch_cuda={torch.version.cuda}"
    )

def run_exp(exp_name: str, exp_config: str,
            run_type: str, nprocesses: int, opts=None, use_navdp: bool = False, use_fmm: bool = True, debug_episodes: str = None, episode_file: str = None, resume_from_log: str = None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()

    # One directory per round. Everything a run emits -- stats, visualisations,
    # rgb frames, habitat logger output -- lands under
    # <EVAL_DATA_ROOT>/<exp_name>/ instead of being scattered across
    # data/checkpoints/, data/logs/eval_results/, data/logs/running_log/ and
    # saved_rgb_images/. Set EVAL_DATA_ROOT "" to get the old layout back.
    eval_data_root = getattr(config, 'EVAL_DATA_ROOT', '')
    # Carry the experiment name explicitly; the output directory basename is stats.
    config.EXP_NAME = exp_name
    if eval_data_root:
        run_dir = os.path.join(eval_data_root, exp_name)
        config.CHECKPOINT_FOLDER = os.path.join(run_dir, 'stats')
        config.EVAL_CKPT_PATH_DIR = os.path.join(run_dir, 'stats')
        config.RESULTS_DIR = os.path.join(run_dir, 'eval_results')
        config.VIDEO_DIR = os.path.join(run_dir, 'video')
        config.TENSORBOARD_DIR = os.path.join(run_dir, 'tensorboard')
        config.DEBUG_LOG_DIR = os.path.join(run_dir, 'debug_logs')
        # Already the final directory: the evaluator appends exp_name only when
        # EVAL_DATA_ROOT is empty.
        config.RGB_SAVE_DIR = os.path.join(run_dir, 'rgb')
        running_log_dir = os.path.join(run_dir, 'logs')
    else:
        config.TENSORBOARD_DIR += exp_name
        config.CHECKPOINT_FOLDER += exp_name
        config.EVAL_CKPT_PATH_DIR += exp_name
        config.RESULTS_DIR += exp_name
        config.VIDEO_DIR += exp_name
        running_log_dir = os.path.join('data', 'logs', 'running_log')
    config.RUNNING_LOG_DIR = running_log_dir
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE
    if use_navdp:
        config.USE_NAVDP = True
    elif 'USE_NAVDP' not in config:
        config.USE_NAVDP = False

    if use_fmm:
        config.USE_FMM = True
    elif 'USE_FMM' not in config:
        config.USE_FMM = False
    
    # --no-fmm sets use_fmm=False; default is True. Always honor the CLI value.
    config.USE_FMM = use_fmm

    config.freeze()
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EVAL_CKPT_PATH_DIR, exist_ok=True)
    os.makedirs(config.RUNNING_LOG_DIR, exist_ok=True)
    logger.add_filehandler(os.path.join(config.RUNNING_LOG_DIR, config.LOG_FILE))
    logger.info(f"hyper parameters:\n{config.EVAL}")
    
    # dataset split, start multi-processes
    gpu_ids = config.TORCH_GPU_IDS
    num_devices = len(gpu_ids)
    logger.info(f'num devices: {num_devices}, gpu_ids: {gpu_ids}, num processes: {nprocesses}')
    
    # Warn if oversubscribing GPUs (recommend at most 2 processes per GPU).
    if nprocesses > num_devices * 2:
        logger.info(f"Warning: {nprocesses} processes on {num_devices} GPUs may cause resource contention")
    
    episode_info_list = get_episode_ids_from_config(config)
    total_available = len(episode_info_list)
    logger.info(f"total available episodes: {total_available}")
    import sys
    sys.stdout.flush()
    
    if episode_file:
        with open(episode_file, 'r') as f:
            fixed_ids = set(json.load(f))
        episode_info_list = [info for info in episode_info_list if info['id'] in fixed_ids]
        logger.info(f"EPISODE FILE: Loaded {len(fixed_ids)} IDs from {episode_file}, matched {len(episode_info_list)}/{total_available}")
    elif debug_episodes:
        debug_ids = [eid.strip() for eid in debug_episodes.split(',')]
        debug_ids_set = set(debug_ids)
        episode_info_list = [info for info in episode_info_list if info['id'] in debug_ids_set]
        logger.info(f"DEBUG MODE: Filtering for specific episodes: {debug_ids}")
        logger.info(f"Found {len(episode_info_list)} matching episodes out of {total_available}")
        if len(episode_info_list) == 0:
             logger.info("Warning: No matching episodes found!")

        config.defrost()
        config.DEBUG_LOGGING = True
        config.DEBUG_EPISODES = debug_ids
        config.freeze()
    else:
        logger.info(f"Using all {total_available} episodes (no sampling)")

    # Resume mode: skip episodes that already appeared in a previous run log.
    # Treat "appeared in log" as "already evaluated" (success=0 is still done).
    if resume_from_log:
        done_ids = extract_episode_ids_from_log(resume_from_log)
        before = len(episode_info_list)
        if done_ids:
            episode_info_list = [info for info in episode_info_list if info["id"] not in done_ids]
        skipped = before - len(episode_info_list)
        logger.info(
            f"RESUME: log={resume_from_log}, found {len(done_ids)} episode tags; "
            f"skipping {skipped}/{before}; remaining {len(episode_info_list)}"
        )

        # When resuming, ignore EPISODE_COUNT truncation unless user explicitly
        # provides debug_episodes.
        config.defrost()
        config.EVAL.EPISODE_COUNT = -1
        config.freeze()

        if len(episode_info_list) == 0:
            logger.info("RESUME: No remaining episodes to process. Exiting.")
            return

        try:
            remaining_ids_path = os.path.join(
                config.RUNNING_LOG_DIR,
                f"remaining_episode_ids_from_{os.path.basename(resume_from_log)}.json",
            )
            os.makedirs(os.path.dirname(remaining_ids_path), exist_ok=True)
            with open(remaining_ids_path, "w") as f:
                json.dump([info["id"] for info in episode_info_list], f)
            logger.info(f"RESUME: wrote remaining episode ids to {remaining_ids_path}")
        except Exception as e:
            logger.info(f"RESUME: failed to write remaining episode ids file: {e}")
    
    if config.EVAL.EPISODE_COUNT > -1 and len(episode_info_list) > config.EVAL.EPISODE_COUNT:
        logger.info(f"Truncating episode list from {len(episode_info_list)} to {config.EVAL.EPISODE_COUNT} based on EVAL.EPISODE_COUNT")
        random.seed(time.time())
        random.shuffle(episode_info_list)
        episode_info_list = episode_info_list[:config.EVAL.EPISODE_COUNT]

    episode_ids_for_print = [info['id'] for info in episode_info_list]
    logger.info(f"Total episodes to process: {len(episode_info_list)}, {episode_ids_for_print}")

    dynamic_queue = bool(getattr(config.EVAL, 'DYNAMIC_QUEUE', False))
    if dynamic_queue:
        logger.info(f"DYNAMIC_QUEUE=True: {nprocesses} workers will pop from a shared queue of {len(episode_info_list)} episodes")
        # Sort by descending bucket size — the LPT (longest-processing-time-first)
        # heuristic: hand out episodes from the largest scenes first so the
        # last-claimed episodes are short ones.
        from collections import Counter
        scene_sizes = Counter(info['scene_id'] for info in episode_info_list)
        episode_info_list.sort(key=lambda x: -scene_sizes[x['scene_id']])

    split_episode_infos = [episode_info_list[i::nprocesses] for i in range(nprocesses)]

    if not dynamic_queue:
        for i, ep_infos in enumerate(split_episode_infos):
            logger.info(f"Process {i}: {len(ep_infos)} episodes")
    
    configs = []
    if dynamic_queue:
        # Build N worker configs with empty EPISODES_ALLOWED; per-ep filter is
        # set inside eval_dynamic() as the worker pops from the queue.
        for i in range(nprocesses):
            shared_config = deepcopy(config)
            shared_config.defrost()
            device_num = gpu_ids[i % num_devices]
            shared_config.local_rank = i
            shared_config.world_size = nprocesses
            shared_config.TORCH_GPU_ID = device_num
            shared_config.TORCH_GPU_IDS = [device_num]
            shared_config.SIMULATOR_GPU_IDS = [device_num]
            shared_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = []
            shared_config.EVAL.EPISODE_COUNT = -1
            shared_config.freeze()
            configs.append(shared_config)
            logger.info(f"Process {i}: GPU {device_num}, dynamic queue")
    else:
        for i, ep_infos in enumerate(split_episode_infos):
            if len(ep_infos) == 0:  # skip processes with no episodes assigned
                logger.info(f"Warning: Process {i} has no episodes assigned, skipping")
                continue

            # Save assignment file for the worker. Namespace by exp_name so
            # parallel runs don't clobber each other's assignments.
            assignment_file = os.path.join(
                config.RUNNING_LOG_DIR, f"worker_{i}_assignments_{exp_name}.json"
            )
            os.makedirs(os.path.dirname(assignment_file), exist_ok=True)
            with open(assignment_file, 'w') as f:
                json.dump(ep_infos, f)

            shared_config = deepcopy(config)
            shared_config.defrost()
            device_num = gpu_ids[i % num_devices]
            shared_config.local_rank = i
            shared_config.world_size = nprocesses
            shared_config.TORCH_GPU_ID = device_num
            shared_config.TORCH_GPU_IDS = [device_num]
            shared_config.SIMULATOR_GPU_IDS = [device_num]
            # Pass IDs for Habitat filtering (may include duplicates)
            shared_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [info['id'] for info in ep_infos]
            shared_config.EVAL.EPISODE_COUNT = -1
            shared_config.freeze()
            configs.append(shared_config)
            logger.info(f"Process {i}: GPU {device_num}, {len(ep_infos)} episodes")

    logger.info(f"Actually starting {len(configs)} processes")

    # Use individual mp.Process per worker instead of Pool so a finished
    # worker's subprocess exits immediately and releases its GPU context
    # (Pool keeps workers alive until the last task finishes, holding memory).
    procs = []
    if dynamic_queue:
        manager = mp.Manager()
        ep_queue = manager.Queue()
        for ep_info in episode_info_list:
            ep_queue.put(ep_info)
        for cfg in configs:
            p = mp.Process(target=worker_dynamic, args=(cfg, ep_queue))
            p.start()
            procs.append(p)
    else:
        for cfg in configs:
            p = mp.Process(target=worker, args=(cfg,))
            p.start()
            procs.append(p)

    logger.info(f"Starting multiprocessing with {len(procs)} workers...")
    start_time = time.time()
    run_timeout = float(getattr(config.EVAL, 'RUN_TIMEOUT_S', 36000))
    timed_out = False

    try:
        while True:
            alive = [p for p in procs if p.is_alive()]
            if not alive:
                break
            elapsed = time.time() - start_time
            logger.info(f"Progress check: {elapsed:.0f}s elapsed, {len(alive)}/{len(procs)} workers still running")
            if elapsed > run_timeout:
                # Terminating workers leaves a partial episode set; report the truncation.
                logger.error(
                    f"TIMEOUT: {elapsed:.0f}s exceeded EVAL.RUN_TIMEOUT_S={run_timeout:.0f}s "
                    f"with {len(alive)}/{len(procs)} workers still running. Terminating "
                    f"them -- RESULTS WILL BE INCOMPLETE."
                )
                for p in alive:
                    p.terminate()
                timed_out = True
                break
            time.sleep(30)

        for p in procs:
            p.join()

        total_time = time.time() - start_time
        successful = sum(1 for p in procs if p.exitcode == 0)
        failed = len(procs) - successful
        logger.info(f"Multiprocessing completed in {total_time:.0f}s")
        logger.info(f"Successful workers: {successful}/{len(procs)}")
        if failed > 0:
            logger.info(f"Failed workers: {failed} (exitcodes: {[p.exitcode for p in procs if p.exitcode != 0]})")
            if timed_out:
                # Every worker we terminated counts as "failed", so raising here
                # would skip aggregation entirely and throw away the stats for the
                # episodes that DID finish. Report and continue instead.
                logger.error(
                    f"{failed}/{len(procs)} workers were terminated by the run timeout; "
                    f"aggregating the episodes that completed before it."
                )
            else:
                raise RuntimeError(f"{failed}/{len(procs)} evaluation workers failed")

    except Exception as e:
        logger.info(f"Error occurred: {e}")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        logger.info("All processes terminated")
        raise
    # Merge and print final statistics
    logger.info("\n" + "="*50)
    logger.info("AGGREGATED STATISTICS ACROSS ALL PROCESSES")
    logger.info("="*50)
    
    # Load and merge episode stats
    from collections import defaultdict
    aggregated_metrics = defaultdict(list)
    total_episodes = 0
    evaluated_episode_ids = []
    
    stats_files = glob.glob(os.path.join(config.EVAL_CKPT_PATH_DIR, "stats_ep_ckpt_*.json"))
    for stats_file in stats_files:
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
                total_episodes += len(data)
                evaluated_episode_ids.extend(list(data.keys()))
                for ep_id, metrics in data.items():
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            aggregated_metrics[k].append(v)
        except Exception as e:
            logger.info(f"Error loading {stats_file}: {e}")
    
    # Calculate averages
    final_averages = {}
    for k, v_list in aggregated_metrics.items():
        if v_list:
            final_averages[k] = sum(v_list) / len(v_list)
    
    logger.info(f"Total Episodes Evaluated: {total_episodes}")
    logger.info(f"Evaluated Episode IDs: {sorted(evaluated_episode_ids)}")
    # Averages over a truncated set are not comparable with a full run, so make the
    # shortfall explicit next to the numbers rather than leaving it to be inferred
    # from a termination message thousands of lines earlier.
    requested = len(episode_info_list)
    if total_episodes < requested:
        missing = sorted(set(str(i['id']) for i in episode_info_list)
                         - set(str(e) for e in evaluated_episode_ids))
        logger.error(
            f"INCOMPLETE: {total_episodes}/{requested} requested episodes produced "
            f"stats. Metrics below are averaged over the completed subset only."
        )
        logger.error(f"INCOMPLETE: missing episode ids: {missing}")
    logger.info("Final Average Metrics:")
    pprint(final_averages)
    
    # Merge model usage stats
    from vlnce_baselines.utils.stats import merge_model_usage_stats
    merge_model_usage_stats(config.EVAL_CKPT_PATH_DIR, config.TASK_CONFIG.DATASET.SPLIT)
    
    # Merge navigation stats
    nav_stats_files = glob.glob(os.path.join(config.EVAL_CKPT_PATH_DIR, "nav_stats_*.json"))
    nav_counter_keys = [
        'total_backtracks',
        'total_waypoints',
        'total_target_set',
        'total_target_timeout',
        'total_projection_pullback',
        'total_stop_requested',
        'total_stop_rejected',
        'total_stop_gate_pass',
        'total_stop_gate_fail',
        'total_stop_forced',
        'total_stop_cooldown_suppressed',
        'total_autostop',
    ]
    nav_totals = {k: 0 for k in nav_counter_keys}

    for fpath in nav_stats_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                for k in nav_counter_keys:
                    nav_totals[k] += data.get(k, 0)
        except:
            pass

    logger.info("\n" + "="*50)
    logger.info("NAVIGATION STATISTICS")
    logger.info("="*50)
    logger.info(f"Total Backtracks: {nav_totals['total_backtracks']}")
    logger.info(f"Total Waypoints: {nav_totals['total_waypoints']}")
    logger.info(f"Targets Set: {nav_totals['total_target_set']}")
    logger.info(f"Target Timeouts: {nav_totals['total_target_timeout']}")
    logger.info(f"Projection Pullbacks: {nav_totals['total_projection_pullback']}")
    logger.info("--- STOP path ---")
    logger.info(f"STOP Requested (double-check): {nav_totals['total_stop_requested']}")
    logger.info(f"STOP Rejected (total): {nav_totals['total_stop_rejected']}")
    logger.info(f"STOP Gate Pass: {nav_totals['total_stop_gate_pass']}")
    logger.info(f"STOP Gate Fail: {nav_totals['total_stop_gate_fail']}")
    logger.info(f"STOP Forced (legacy only): {nav_totals['total_stop_forced']}")
    logger.info(f"STOP Suppressed by cooldown: {nav_totals['total_stop_cooldown_suppressed']}")
    logger.info(f"Budget Autostops: {nav_totals['total_autostop']}")
    logger.info("="*50)

def worker(config):
    try:
        worker_log_file = os.path.join(
            config.RUNNING_LOG_DIR, f"worker_{config.local_rank}_{config.LOG_FILE}"
        )
        os.makedirs(os.path.dirname(worker_log_file), exist_ok=True)
        logger.add_filehandler(worker_log_file)

        logger.info(f"Worker started: local_rank={config.local_rank}, device={config.TORCH_GPU_ID}")
        import sys
        sys.stdout.flush()
        logger.info(f"Worker {config.local_rank} started on GPU {config.TORCH_GPU_ID}")
        logger.info(f"Worker {config.local_rank} processing {len(config.TASK_CONFIG.DATASET.EPISODES_ALLOWED)} episodes")

        register_extensions()
        seed_everything(config.TASK_CONFIG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.set_num_threads(1)

        TRAINER = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert TRAINER is not None, f"{config.TRAINER_NAME} is not supported"
        logger.info(f"Worker {config.local_rank}: Starting trainer")
        trainer = TRAINER(config)
        trainer.eval()
        logger.info(f"Worker {config.local_rank}: Completed successfully")
        return True
    except Exception as e:
        logger.info(f"Worker {config.local_rank} failed with error: {str(e)}")
        logger.error(f"Worker {config.local_rank} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())
        raise


def worker_dynamic(config, ep_queue):
    """Worker entry for EVAL.DYNAMIC_QUEUE: pop episodes from a shared queue.

    Builds the trainer once (heavy: GroundedSAM, RepViTSAM, policy, agent),
    then loops on the queue rebuilding only the habitat env per ep.
    """
    try:
        worker_log_file = os.path.join(
            config.RUNNING_LOG_DIR, f"worker_{config.local_rank}_{config.LOG_FILE}"
        )
        os.makedirs(os.path.dirname(worker_log_file), exist_ok=True)
        logger.add_filehandler(worker_log_file)

        logger.info(f"Worker(dyn) started: local_rank={config.local_rank}, device={config.TORCH_GPU_ID}")
        import sys
        sys.stdout.flush()
        logger.info(f"Worker {config.local_rank} (dyn) started on GPU {config.TORCH_GPU_ID}")

        register_extensions()
        seed_everything(config.TASK_CONFIG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.set_num_threads(1)

        TRAINER = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert TRAINER is not None, f"{config.TRAINER_NAME} is not supported"
        trainer = TRAINER(config)
        trainer.eval_dynamic(ep_queue)
        logger.info(f"Worker {config.local_rank} (dyn): Completed successfully")
        return True
    except Exception as e:
        logger.info(f"Worker {config.local_rank} (dyn) failed: {e}")
        logger.error(f"Worker {config.local_rank} (dyn) failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default="test",
        required=True,
        help="unique experiment name for output and resume bookkeeping",
    )
    parser.add_argument(
        "--run-type",
        choices=["eval"],
        required=True,
        help="run type of the experiment(train, eval, inference), only eval for zero-shot vln",
    )
    parser.add_argument(
        "--nprocesses",
        type=int,
        default=1,
        help="number of processes",
    )
    parser.add_argument(
        "--use-navdp",
        action="store_true",
        default=False,
        help="Use NavDP for trajectory planning (default: False)",
    )
    parser.add_argument(
        "--use-fmm",
        action="store_true",
        default=True,
        help="Use FMM for trajectory planning (default: True)",
    )
    parser.add_argument(
        "--no-fmm",
        action="store_false",
        dest="use_fmm",
        help="Disable FMM",
    )
    parser.add_argument(
        "--debug-episodes",
        type=str,
        default=None,
        help="Comma-separated list of episode IDs to run for debugging (e.g., '232,330'). Overrides other sampling options.",
    )
    parser.add_argument(
        "--episode-file",
        type=str,
        default=None,
        help="Path to JSON file containing list of episode IDs to evaluate (e.g., joint_stratified_100_seed42.json).",
    )
    parser.add_argument(
        "--resume-from-log",
        type=str,
        default=None,
        help=(
            "Resume eval by skipping episodes already present in a previous log. "
            "The log should contain lines like 'ep187:{...}'. "
            "Example: --resume-from-log logs/0310-165718.log"
        ),
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    logger.info(args)

    validate_model_environment()
    validate_cuda_environment()
    mp.set_start_method('spawn', force=True)
    run_exp(**vars(args))
