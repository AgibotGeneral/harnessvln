# Shared engine for the per-dataset eval scripts. Source this at the end of a
# dataset script after setting DATASET / EXP_CONFIG (and optionally
# DEFAULT_TIMEOUT_S), then call run_eval "$@".
#
# Knobs (all optional, all env vars):
#   GPUS=0,1,2,3        physical cards to use. NPROC defaults to 8 per card.
#   NPROC=32            worker processes, handed round-robin over the cards.
#   TIMESTAMP=<name>    exp name; defaults to MMDD-HHMMSS. Also names the run
#                       directory eval_data/<name>/.
#   EVAL_DATA_ROOT=...  where run directories go. Default eval_data.
#   EPISODES=full       full | sample | <path to a JSON list of episode ids>
#   RESUME_FROM=a,b     skip episodes already completed in exp names a and b.
#   HARNESS=True        EVAL.HARNESS_ENABLED.
#   RUN_TIMEOUT_S=...   wall-clock ceiling; workers are killed past it.
#   PYTHON=<interp>     interpreter. Must be the habitat env, not python3.
#   MAGNUM_LOG=quiet    habitat-sim verbosity. verbose blows up the log size.
#   DRY_RUN=1           print the command and exit.
#   FORCE=1             allow reusing an exp name that already has data.
#
# Everything a round produces lands under eval_data/<exp>/:
#   stats/         per-worker stats_ep_ckpt_* / nav_stats_* / model_usage_*
#   eval_results/  fmm_fields + waypoints visualisations
#   rgb/           per-episode frames and events_*.jsonl
#   logs/          console.log, main/worker habitat logs, resume artifacts
#
# Launcher flags and configuration overrides are separated before invocation.
# Explicit configuration overrides follow defaults, so caller values win.

set -o pipefail

_die() { echo "ERROR: $*" >&2; exit 1; }

# Union of the episode ids a previous run finished, taken from BOTH the stats
# checkpoints and the run log.
#
# Neither source alone is sufficient. The stats JSONs are the real record but
# only name episodes whose worker flushed; the log's `ep<id>:` tags are what
# run_mp.py --resume-from-log actually parses but a log can be truncated or
# rotated. Emitting the union as a synthetic log gives --resume-from-log a
# complete and stable input.
_build_done_list() {
    local out_path="$1"; shift
    EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-eval_data}" "${PYTHON}" - "${out_path}" "$@" <<'PY'
import glob, json, os, re, sys

root = os.environ.get("EVAL_DATA_ROOT", "eval_data")
out_path, exps = sys.argv[1], sys.argv[2:]
done, per_exp = set(), []
tag = re.compile(r"\bep(\d+)\s*:")

for exp in exps:
    before = len(done)
    # New layout first, then the pre-eval_data one. Rounds from before the
    # move still live under data/checkpoints/<exp>/ and must stay resumable.
    ckpt_dirs = [os.path.join(root, exp, "stats"),
                 os.path.join("data", "checkpoints", exp)]
    n_files = 0
    for ckpt_dir in ckpt_dirs:
        for f in glob.glob(os.path.join(ckpt_dir, "stats_ep_ckpt_*.json")):
            n_files += 1
            try:
                done.update(str(k) for k in json.load(open(f)))
            except Exception as e:
                print(f"  WARNING: unreadable {f}: {e}", file=sys.stderr)
    # Prefer the frozen copy: a rerun under the same exp name would have
    # truncated the live log via tee.
    logs = [os.path.join(root, exp, "logs", "console.resume-src.log"),
            os.path.join(root, exp, "logs", "console.log"),
            f"logs/{exp}.resume-src.log",
            f"logs/{exp}.log"]
    log_used = None
    for lg in logs:
        if os.path.exists(lg):
            log_used = lg
            with open(lg, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = tag.search(line)
                    if m:
                        done.add(m.group(1))
            break
    if not n_files and log_used is None:
        raise SystemExit(
            f"RESUME: '{exp}' has no stats under {ckpt_dirs[0]}/ or {ckpt_dirs[1]}/ "
            f"and no console log. Refusing to resume from an exp name that does not "
            f"exist -- a typo here silently reruns the whole set."
        )
    per_exp.append((exp, n_files, log_used, len(done) - before))

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w") as f:
    for ep in sorted(done, key=lambda x: (len(x), x)):
        f.write(f"ep{ep}:{{}}\n")

for exp, n_files, log_used, added in per_exp:
    print(f"  {exp}: {n_files} stats files, log={log_used or 'none'}, +{added} new ids")
print(f"  total distinct completed episodes: {len(done)}")
PY
}

run_eval() {
    : "${DATASET:?dataset script must set DATASET}"
    : "${EXP_CONFIG:?dataset script must set EXP_CONFIG}"

    local repo_dir
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "${repo_dir}" || _die "cannot cd to ${repo_dir}"

    local gpus="${GPUS:-0,1,2,3}"
    local ngpu
    ngpu=$(awk -F, '{print NF}' <<< "${gpus}")
    [ "${ngpu}" -ge 1 ] || _die "GPUS is empty"

    # 8 workers per card is the measured ceiling: ~3.2 GiB each leaves ~6 GiB
    # headroom on a 32.6 GiB card. Going higher risks a mid-run OOM.
    local nproc="${NPROC:-$((ngpu * 8))}"

    # TORCH_GPU_IDS must be the *visible* indices (0..N-1 after the
    # CUDA_VISIBLE_DEVICES remapping), not the physical ids in GPUS.
    local gpu_list="[$(seq -s, 0 $((ngpu - 1)))]"

    local timestamp="${TIMESTAMP:-$(date +"%m%d-%H%M%S")}"
    source "${repo_dir}/eval_scripts/lib/runtime_env.sh"
    resolve_python || exit 1

    # One directory per round: stats/, eval_results/, rgb/, logs/. run_mp.py
    # derives the same layout from the EVAL_DATA_ROOT config key, so the
    # launcher's own console log lands next to what the workers write.
    local eval_root="${EVAL_DATA_ROOT:-eval_data}"
    local run_dir="${eval_root}/${timestamp}"
    local log_file="${run_dir}/logs/console.log"
    local ckpt_dir="${run_dir}/stats"
    if [ -z "${DRY_RUN:-}" ]; then
        mkdir -p "${run_dir}/logs"
    fi

    # Guard the two ways a rerun silently destroys a previous round's data:
    #   1. the launcher ends in `tee logs/<exp>.log`, which TRUNCATES the log
    #      that --resume-from-log needs to read;
    #   2. each worker starts from an empty state_eps and writes back to the
    #      same stats_ep_ckpt_<split>_r<rank>_w<n>.json, overwriting the rows
    #      a previous run under this name had already produced.
    # Resuming must therefore go to a NEW exp name.
    if [ -z "${FORCE:-}" ]; then
        local existing=""
        [ -s "${log_file}" ] && existing="${log_file}"
        if compgen -G "${ckpt_dir}/stats_ep_ckpt_*.json" > /dev/null; then
            existing="${existing:+${existing} and }${ckpt_dir}/"
        fi
        # A round from before the eval_data move still counts as a collision.
        if compgen -G "data/checkpoints/${timestamp}/stats_ep_ckpt_*.json" > /dev/null; then
            existing="${existing:+${existing} and }data/checkpoints/${timestamp}/"
        fi
        if [ -n "${existing}" ]; then
            _die "exp name '${timestamp}' already has data (${existing}).
       Reusing it would truncate that log and overwrite those stats.
       Use a new TIMESTAMP (and RESUME_FROM=${timestamp} to skip what it finished),
       or set FORCE=1 if you really mean to overwrite."
        fi
    fi

    export CUDA_VISIBLE_DEVICES="${gpus}"
    export HF_HUB_OFFLINE=1
    export BERT_LOCAL_PATH=data/grounded_sam/bert-base-uncased
    export TOKENIZERS_PARALLELISM=false
    export GLOG_minloglevel=0
    # Keep simulator logging quiet by default.
    export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
    # EGL_PLATFORM / DISPLAY and the container EGL workaround live here.
    if [ -z "${DRY_RUN:-}" ]; then
        source "${repo_dir}/eval_scripts/egl_env.sh"
    fi

    local -a flags=(
        --exp-name "${timestamp}"
        --run-type eval
        --exp-config "${EXP_CONFIG}"
        --nprocesses "${nproc}"
        --use-navdp
    )

    # Episode selection. Default is the full val set; a sample must be asked
    # for, so a round is never quietly 100 episodes when it looks like 3000.
    local episodes="${EPISODES:-full}"
    local ep_desc
    case "${episodes}" in
        full)
            ep_desc="FULL ${DATASET} val set (no --episode-file)"
            ;;
        sample)
            local sample_file="data/datasets/stratified_samples/${DATASET}.json"
            [ -f "${sample_file}" ] || _die "no stratified sample at ${sample_file}"
            flags+=(--episode-file "${sample_file}")
            ep_desc="stratified sample ${sample_file}"
            ;;
        *)
            [ -f "${episodes}" ] || _die "EPISODES='${episodes}' is neither 'full', 'sample', nor an existing file"
            flags+=(--episode-file "${episodes}")
            ep_desc="custom episode file ${episodes}"
            ;;
    esac

    # Resume. Build the done-list first so a bad exp name fails here, before
    # 32 workers and an hour of habitat startup.
    local resume_desc="none"
    if [ -n "${RESUME_FROM:-}" ]; then
        local done_list="${run_dir}/logs/resume-from.log"
        echo "Resume: reading completed episodes from '${RESUME_FROM}'"
        # shellcheck disable=SC2086
        EVAL_DATA_ROOT="${eval_root}" _build_done_list "${done_list}" ${RESUME_FROM//,/ } || exit 1
        flags+=(--resume-from-log "${done_list}")
        resume_desc="${RESUME_FROM} (done-list: ${done_list})"
    fi

    local -a opts=(
        NUM_ENVIRONMENTS 1
        TRAINER_NAME ZS-Evaluator-mp
        EVAL_DATA_ROOT "${eval_root}"
        TORCH_GPU_IDS "${gpu_list}"
        SIMULATOR_GPU_IDS "${gpu_list}"
        EVAL.HARNESS_ENABLED "${HARNESS:-True}"
        EVAL.RUN_TIMEOUT_S "${RUN_TIMEOUT_S:-${DEFAULT_TIMEOUT_S:-259200}}"
    )

    # EVAL.EPISODE_COUNT defaults to 5000 and run_mp.py SHUFFLES then truncates
    # to it, so a full set larger than that (rxr has 11006) would silently
    # become a random 5000-episode subset. -1 disables the truncation. Only do
    # this for full runs -- an explicit episode file is already the selection.
    if [ "${episodes}" = "full" ]; then
        opts+=(EVAL.EPISODE_COUNT -1)
    fi

    echo "=============================================================="
    echo "  dataset   : ${DATASET}  (${EXP_CONFIG})"
    echo "  exp name  : ${timestamp}"
    echo "  episodes  : ${ep_desc}"
    echo "  resume    : ${resume_desc}"
    echo "  GPUs      : ${gpus} -> visible ${gpu_list}"
    echo "  workers   : ${nproc}  ($((nproc / ngpu)) per card)"
    echo "  harness   : ${HARNESS:-True}"
    echo "  timeout   : ${RUN_TIMEOUT_S:-${DEFAULT_TIMEOUT_S:-259200}}s"
    echo "  python    : ${PYTHON}"
    echo "  run dir   : ${run_dir}/"
    echo "  log       : ${log_file}"
    echo "  stats     : ${ckpt_dir}/"
    echo "=============================================================="

    # Keep launcher flags before argparse's configuration REMAINDER. Apply
    # user config overrides after defaults so explicit values win.
    local -a cli_args=() config_overrides=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --use-navdp|--use-fmm|--no-fmm|--help|-h)
                cli_args+=("$1"); shift ;;
            --exp-name|--run-type|--nprocesses|--debug-episodes|--episode-file|--resume-from-log|--exp-config)
                [ "$#" -ge 2 ] || _die "missing value for $1"
                cli_args+=("$1" "$2"); shift 2 ;;
            --*=*) cli_args+=("$1"); shift ;;
            --) shift ;;
            --*) _die "unknown launcher argument $1" ;;
            *)
                [ "$#" -ge 2 ] || _die "configuration override $1 needs a value"
                config_overrides+=("$1" "$2"); shift 2 ;;
        esac
    done
    if [ -n "${DRY_RUN:-}" ]; then
        echo "DRY_RUN: would run"
        echo "  ${PYTHON} run_mp.py ${flags[*]} ${cli_args[*]} ${opts[*]} ${config_overrides[*]}"
        exit 0
    fi

    "${PYTHON}" run_mp.py "${flags[@]}" "${cli_args[@]}" "${opts[@]}" "${config_overrides[@]}" 2>&1 | tee "${log_file}"
    local status=${PIPESTATUS[0]}

    # Freeze the log the moment the run ends. This copy is what a later
    # RESUME_FROM reads, and unlike console.log nothing will tee over it.
    if [ -s "${log_file}" ]; then
        local frozen="${run_dir}/logs/console.resume-src.log"
        rm -f "${frozen}" 2>/dev/null
        cp -a "${log_file}" "${frozen}" 2>/dev/null \
            && chmod a-w "${frozen}" 2>/dev/null
        echo "Froze resume source: ${frozen}"
    fi

    echo "Run '${timestamp}' exited with status ${status}"
    return "${status}"
}
