from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import Config as CN
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat_extensions.config.default import get_extended_config as get_task_config


# Experiment configuration.
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/zs_vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "ZS-Evaluator"
_C.ENV_NAME = "VLNCEZeroShotEnv"
_C.SIMULATOR_GPU_IDS = [0]
# Group run artifacts under <EVAL_DATA_ROOT>/<exp_name>.
# An empty root uses the individual output paths instead.
_C.EVAL_DATA_ROOT = "eval_data"
# Set from --exp-name by the launcher.
_C.EXP_NAME = ""
# Overridden by the launcher when EVAL_DATA_ROOT is set.
_C.RUNNING_LOG_DIR = "data/logs/running_log"
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"
_C.BLIP2_MODEL_DIR = "data/blip2/blip2_model.pt"
_C.BLIP2_VIS_PROCESSORS_DIR = "data/blip2/blip2_vis_processors.pt"
_C.BLIP2_TEXT_PROCESSORS_DIR = "data/blip2/blip2_text_processors.pt"
_C.VQA_MODEL_DIR = "data/vqa/vqa_model.pt"
_C.VQA_VIS_PROCESSORS_DIR = "data/vqa/vqa_vis_processors.pt"
_C.VQA_TEXT_PROCESSORS_DIR = "data/vqa/vqa_text_processors.pt"
_C.KEYBOARD_CONTROL = 0
_C.SAVE_RGB_IMAGES = False
_C.RGB_SAVE_DIR = "saved_rgb_images"
_C.DEBUG_LOGGING = False
_C.DEBUG_LOG_DIR = "logs/debug_logs"
_C.TASK_TYPE = "VLN"  # one of {"VLN", "OBJNAV"}

# Mapping configuration.
_C.MAP = CN()
_C.MAP.GROUNDING_DINO_CONFIG_PATH = "data/grounded_sam/GroundingDINO_SwinT_OGC.py"
_C.MAP.GROUNDING_DINO_CHECKPOINT_PATH = "data/grounded_sam/groundingdino_swint_ogc.pth"
_C.MAP.SAM_CHECKPOINT_PATH = "data/grounded_sam/sam_vit_h_4b8939.pth"
_C.MAP.RepViTSAM_CHECKPOINT_PATH = "data/grounded_sam/repvit_sam.pt"
_C.MAP.SAM_ENCODER_VERSION = "vit_h"
_C.MAP.BOX_THRESHOLD = 0.25
_C.MAP.TEXT_THRESHOLD = 0.25
_C.MAP.FRAME_WIDTH = 160
_C.MAP.FRAME_HEIGHT = 120
_C.MAP.MAP_RESOLUTION = 5
_C.MAP.MAP_SIZE_CM = 2400
_C.MAP.GLOBAL_DOWNSCALING = 2
_C.MAP.VISION_RANGE = 100
_C.MAP.DU_SCALE = 1
_C.MAP.CAT_PRED_THRESHOLD = 5.0
_C.MAP.EXP_PRED_THRESHOLD = 1.0
_C.MAP.MAP_PRED_THRESHOLD = 1.0
_C.MAP.MAX_SEM_CATEGORIES = 16
_C.MAP.CENTER_RESET_STEPS = 25
_C.MAP.MIN_Z = 2  # Height cutoff in centimetres; retain near-floor geometry.
_C.MAP.VISUALIZE = False
_C.MAP.PRINT_IMAGES = False
_C.MAP.REPVITSAM = 0


# Evaluation configuration.
_C.EVAL = CN()
_C.EVAL.SPLIT = "val_unseen"  # The split to evaluate on
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.EPISODE_COUNT = 5000
_C.EVAL.SAVE_RESULTS = True
# Enable TODO memory.
_C.EVAL.USE_TODO_LIST = True
# Allow replanning at a backtrack waypoint.
_C.EVAL.BACKTRACK_SECOND_CHANCE = True
# Workers share an episode queue and reuse their trainer/model across episodes.
_C.EVAL.DYNAMIC_QUEUE = True
# Sample forward-view RGB frames throughout navigation.
_C.EVAL.USE_CONTINUOUS_HISTORY = True
# Simulator steps between history frames.
_C.EVAL.HISTORY_INTERVAL = 2
# Thin older history frames to keep requests within the provider image limit.
_C.EVAL.MAX_IMAGES_PER_REQUEST = 50
# Wall-clock limit in seconds; unfinished workers are terminated at expiry.
_C.EVAL.RUN_TIMEOUT_S = 36000  # 10h

# STOP geometry and budget verification (evidence-based stop validation).
# Off by default: the model's double-checked STOP executes as requested. Enable
# STOP_GEOMETRIC_GATE and STOP_AUTOSTOP_BEFORE_TIMEOUT together, with a
# 24-step STOP_REJECTION_PANORAMA_COOLDOWN_STEPS, to turn it on.

# Legacy STOP behavior: no geometric gate or budget autostop; force STOP after three rejections.
_C.EVAL.LEGACY_STOP_ENABLED = False
# Require geometry to pass before executing a model-requested STOP.
_C.EVAL.STOP_GEOMETRIC_GATE = False
# Map-target distance threshold in metres, independent of TASK.SUCCESS_DISTANCE.
_C.EVAL.STOP_GEOMETRIC_GATE_DISTANCE_M = 1.0
# Maximum target age in simulator steps; includes approach and verification panoramas.
_C.EVAL.STOP_GEOMETRIC_GATE_MAX_TARGET_AGE_STEPS = 150
# Use reserved steps for a geometrically verified STOP before timeout.
_C.EVAL.STOP_AUTOSTOP_BEFORE_TIMEOUT = False
# The reserve must cover the largest loop increment (13) and two attempts (26).
# 38 = 13 - 1 + 26 leaves both attempts affordable at the latest entry.
_C.EVAL.STOP_AUTOSTOP_RESERVE_STEPS = 38
# Each attempt requires its own panorama and STOP budget.
_C.EVAL.STOP_AUTOSTOP_MAX_ATTEMPTS = 2
# fresh_depth measures the current object surface; map_target uses the stored projection.
# Stored targets can be displaced by projection pullback or cached observations.
_C.EVAL.STOP_GATE_DISTANCE_BASIS = 'fresh_depth'
# Object-surface depth needs a separate threshold from map-target distance.
# Goal viewpoints can stand back from the surface.
_C.EVAL.STOP_GATE_DEPTH_DISTANCE_M = 2.5
# Side length in pixels of the median-depth patch at the bbox anchor.
_C.EVAL.STOP_GATE_DEPTH_WINDOW = 5
# Reject patches with depth standard deviation above this threshold in metres.
_C.EVAL.STOP_GATE_DEPTH_MAX_STD_M = 0.5
# Budget autostop may override a negative visual verdict when geometry passes.
# These events are tagged vlm_declined=true.
_C.EVAL.STOP_AUTOSTOP_ALLOW_VLM_OVERRIDE = True
# Write per-episode target, STOP and terminal events.
_C.EVAL.STOP_EVENT_STREAM = True
# Flush every N events; 1 flushes each event.
_C.EVAL.STOP_EVENT_FLUSH_EVERY = 1
# Suppress repeated STOP intents after rejection to avoid spending the budget on panoramas.
# 0 disables the cooldown; use 24 with the geometric gate enabled.
_C.EVAL.STOP_REJECTION_PANORAMA_COOLDOWN_STEPS = 0
# Sampling temperature for visual STOP verification across task types.
_C.EVAL.STOP_CHECK_TEMPERATURE = 0.7

# Runtime contracts and structured traces.
_C.EVAL.HARNESS_ENABLED = False
_C.EVAL.HARNESS_TRACE = True
_C.EVAL.HARNESS_STRICT_RECEIPTS = True
_C.EVAL.HARNESS_REQUIRE_FRESH_OBSERVATION = True
_C.EVAL.HARNESS_SAFE_PARSE_FAILURE_ACTION = 'observe'
# Emit a HARNESS_PHASE record on every state-machine transition.
_C.EVAL.HARNESS_TRACE_PHASES = False

# Require completion evidence to cite an observation or receipt issued in this episode.
_C.EVAL.TODO_EVIDENCE_STRICT = False
# Record places, objects and failure history for retrieval.
_C.EVAL.WORLD_GRAPH_ENABLED = True
# Place merge radius in metres; matches the target-reached threshold.
_C.EVAL.WORLD_GRAPH_MERGE_RADIUS_M = 0.75
_C.EVAL.WORLD_GRAPH_OBJECT_MERGE_RADIUS_M = 1.0
# Merge nearly coincident groundings even when free-text object names differ.
_C.EVAL.WORLD_GRAPH_SAME_POINT_RADIUS_M = 0.35
# Include recalled places and prior failures in planner prompts.
_C.EVAL.MEMORY_READOUT_ENABLED = True
# Rank places, expand one hop from these seeds and the current place, then cap the readout.
_C.EVAL.MEMORY_READOUT_SEEDS = 3
_C.EVAL.MEMORY_READOUT_MAX_PLACES = 5
# score = relevance + RECENCY * recency + SALIENCE * salience - penalty
_C.EVAL.MEMORY_RECENCY_WEIGHT = 0.3
_C.EVAL.MEMORY_SALIENCE_WEIGHT = 0.3
# Per-failure penalty for the same subgoal; capped at 1.0 so retries remain possible.
_C.EVAL.MEMORY_FAILURE_PENALTY_PER = 0.5
# Shared distance limit for planner actions and backtrack IDs in memory readouts.
_C.EVAL.MAX_BACKTRACK_DISTANCE_M = 6.0


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    assert config.TASK_TYPE in {"VLN", "OBJNAV"}, (
        f"Unknown TASK_TYPE: {config.TASK_TYPE!r}. Expected one of VLN, OBJNAV."
    )

    config.freeze()
    return config
