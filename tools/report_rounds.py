#!/usr/bin/env python
"""Build a comparison report for two eval rounds and print it as Markdown.

Reads the per-worker `stats_ep_ckpt_*.json` files each worker writes under
eval_data/<exp_name>/stats/ (or the legacy data/checkpoints/<exp_name>/) and the round's console log for error counts, then emits a
Markdown summary for local analysis.

Usage:
    python tools/report_rounds.py run_a run_b
    python tools/report_rounds.py --out report.md run_a run_b
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlnce_baselines.usage import merge_usage

CKPT_ROOT = "data/checkpoints"
LOG_DIR = "logs"
EVAL_DATA_ROOT = os.environ.get("EVAL_DATA_ROOT", "eval_data")


# Resolve the current output layout with a fallback to legacy directories.
def _first_existing(candidates, default):
    for path in candidates:
        if os.path.exists(path):
            return path
    return default


def stats_dir(exp_name):
    new = os.path.join(EVAL_DATA_ROOT, exp_name, "stats")
    return _first_existing([new, os.path.join(CKPT_ROOT, exp_name)], new)


def rgb_dir(exp_name):
    new = os.path.join(EVAL_DATA_ROOT, exp_name, "rgb")
    return _first_existing([new, os.path.join("saved_rgb_images", exp_name)], new)


def console_log(exp_name):
    new = os.path.join(EVAL_DATA_ROOT, exp_name, "logs", "console.log")
    return _first_existing([new, os.path.join(LOG_DIR, f"{exp_name}.log")], new)


# Metrics to report, in display order: (key, label, is_percent)
METRICS = [
    ("success", "SR (success rate)", True),
    ("spl", "SPL", True),
    ("oracle_success", "Oracle SR", True),
    ("distance_to_goal", "Final distance to goal (m)", False),
    ("steps_taken", "Mean steps", False),
    ("path_length", "Mean path length (m)", False),
]

# Episodes where the agent ends on a navmesh island disconnected from the goal
# report distance_to_goal=inf and spl=nan. Averaging those in makes the whole
# column read "inf" / "nan". Exclude them from means and count them separately.
def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean_finite(values):
    vals = [v for v in values if _finite(v)]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def load_round(exp_name):
    """Aggregate one round's per-episode stats. Returns dict with per-metric means."""
    ckpt_dir = stats_dir(exp_name)
    stats_files = sorted(glob.glob(os.path.join(ckpt_dir, "stats_ep_ckpt_*.json")))

    episodes = {}
    per_worker = {}
    for path in stats_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:  # a worker killed mid-write leaves truncated JSON
            per_worker[os.path.basename(path)] = f"READ FAILED: {e}"
            continue
        per_worker[os.path.basename(path)] = len(data)
        # Episode ids are unique across workers (each pops from a shared queue),
        # so a plain update is safe; last-write-wins if a retry duplicated one.
        episodes.update(data)

    means = {}
    counted = {}
    for key, _, _ in METRICS:
        means[key], counted[key] = mean_finite(m.get(key) for m in episodes.values())

    # Non-finite distance_to_goal is the navmesh-island signature; report the
    # count so a shrinking denominator is visible rather than silent.
    non_finite = sorted(
        (ep for ep, m in episodes.items() if not _finite(m.get("distance_to_goal"))),
        key=lambda x: int(x) if str(x).isdigit() else 0,
    )

    return {
        "exp_name": exp_name,
        "ckpt_dir": ckpt_dir,
        "n_episodes": len(episodes),
        "episode_ids": sorted(episodes, key=lambda x: int(x) if str(x).isdigit() else 0),
        "means": means,
        "counted": counted,
        "non_finite": non_finite,
        "buckets": failure_buckets(episodes),
        "per_worker": per_worker,
        "episodes": episodes,
        "usage": load_usage(ckpt_dir),
    }


def step_cap(episodes):
    """Infer the round's MAX_EPISODE_STEPS from the observed step counts.

    Legacy stats omit the configured cap. Round the observed maximum up to
    a multiple of 500, with 500 as the minimum. This is a heuristic; short
    runs cannot establish a larger configured budget from observations alone.
    """
    observed = max((m.get("steps_taken", 0) or 0) for m in episodes.values()) if episodes else 0
    return max(500, int(math.ceil(observed / 500.0) * 500))


def failure_buckets(episodes):
    """Classify unsuccessful episodes by budget, oracle success and final distance.

    "Budget exhausted" is cap-4% rather than the cap itself, approximating
    "never sent STOP": a single decision round can span multiple steps, so an
    episode stops a little short of the ceiling.
    """
    cutoff = step_cap(episodes) * 0.96
    b = {k: [] for k in ("T1", "T2", "T3", "T4", "T5", "T6")}
    for ep, m in episodes.items():
        if m.get("success") == 1.0:
            continue
        steps = m.get("steps_taken", 0)
        oracle = m.get("oracle_success", 0)
        d = m.get("distance_to_goal")
        d = d if _finite(d) else float("inf")
        if steps >= cutoff:
            if oracle == 1 and d <= 1.0:
                b["T1"].append(ep)
            elif oracle == 1:
                b["T2"].append(ep)
            else:
                b["T3"].append(ep)
        else:
            if oracle == 1:
                b["T4"].append(ep)
            elif d <= 2.0:
                b["T5"].append(ep)
            else:
                b["T6"].append(ep)
    return b


def load_usage(ckpt_dir):
    """Sum LLM call/token counts across per-worker model-usage files, if present."""
    files = glob.glob(os.path.join(ckpt_dir, "model_usage_stats_*_r*_w*.json"))
    if not files:
        return None
    records = []
    for path in files:
        with open(path) as stream:
            records.append(json.load(stream))
    merged = merge_usage(records)
    return {'calls': merged['total_calls'], 'total_tokens': merged['total_tokens']}


def log_diagnostics(exp_name):
    """Count API failures, wall-clock, and the models used, from the round's log."""
    path = console_log(exp_name)
    if not os.path.exists(path):
        return {"log": None}
    api_rejected = 0
    api_errors = 0
    retries = 0
    degraded = 0
    last_elapsed = None
    nproc = None
    truncated = False
    model_name = None
    with open(path, errors="replace") as f:
        for line in f:
            if "API Call error" in line:
                api_errors += 1
            elif "API rejected request" in line:
                api_rejected += 1
            elif "API retry " in line:
                retries += 1
            elif "blind 'forward'" in line or "no usable" in line:
                degraded += 1
            elif "Progress check:" in line:
                m = re.search(r"Progress check: (\d+)s elapsed", line)
                if m:
                    last_elapsed = int(m.group(1))
            elif "TIMEOUT:" in line or "Timeout reached" in line:
                truncated = True
            elif model_name is None and "Model: " in line:
                m = re.search(r"Model: (\S+)", line)
                if m:
                    model_name = m.group(1)
            elif nproc is None and "nprocesses=" in line:
                m = re.search(r"nprocesses=(\d+)", line)
                if m:
                    nproc = int(m.group(1))
    return {
        "log": path,
        "api_errors": api_errors,
        "api_rejected": api_rejected,
        "retries": retries,
        "degraded": degraded,
        "elapsed_s": last_elapsed,
        "nprocesses": nproc,
        "truncated": truncated,
        "model_name": model_name,
    }


def harness_diagnostics(exp_name):
    root = rgb_dir(exp_name)
    counts = {
        'HARNESS_TOOL_CALL': 0,
        'HARNESS_TOOL_RESULT': 0,
        'HARNESS_RECEIPT': 0,
        'HARNESS_VIOLATION': 0,
        'HARNESS_TERMINAL': 0,
        'HARNESS_EPISODE_START': 0,
        # Report-only measurements; absent events count as zero.
        'HARNESS_NOTE': 0,
        # STOP-path event counters.
        'AUTOSTOP': 0,
        'AUTOSTOP_SKIPPED': 0,
        'AUTOSTOP_UNAFFORDABLE': 0,
        'STOP_GATE_PASS': 0,
        'STOP_GATE_FAIL': 0,
        'STOP_REJECTED': 0,
        'STOP_FORCED': 0,
        # Track graph writes separately from memory readouts delivered to the planner.
        'WORLD_GRAPH_PLACE': 0,
        'WORLD_GRAPH_OBJECT': 0,
        'WORLD_GRAPH_FAILURE': 0,
        'MEMORY_READOUT': 0,
    }
    # Record whether autostop used fresh depth or fell back to a stored target.
    counts['_autostop_basis'] = {}
    # HARNESS_NOTE / HARNESS_VIOLATION broken down by `kind`, so a note's rate
    # is readable without grepping the traces by hand.
    counts['_note_kinds'] = {}
    if not os.path.isdir(root):
        return counts
    for path in glob.glob(os.path.join(root, '*', 'events_*.jsonl')):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    event = rec.get('event')
                    if event in counts:
                        counts[event] += 1
                    if event in ('HARNESS_NOTE', 'HARNESS_VIOLATION'):
                        key = f"{event}:{rec.get('kind') or 'unrecorded'}"
                        counts['_note_kinds'][key] = \
                            counts['_note_kinds'].get(key, 0) + 1
                    if event in ('AUTOSTOP', 'AUTOSTOP_SKIPPED'):
                        # Events without a distance-basis field have no
                        # distance_basis field at all; label those rather than
                        # emitting a bare "None" that reads like a bug.
                        basis = rec.get('distance_basis') or 'unrecorded'
                        key = f"{event}:{basis}"
                        counts['_autostop_basis'][key] = \
                            counts['_autostop_basis'].get(key, 0) + 1
        except Exception:
            continue
    return counts


def fmt(value, is_percent):
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%" if is_percent else f"{value:.2f}"


def fmt_delta(a, b, is_percent, higher_is_better):
    """Format b - a with an arrow indicating whether it moved the right way."""
    if a is None or b is None:
        return "n/a"
    d = b - a
    body = f"{d * 100:+.1f}pp" if is_percent else f"{d:+.2f}"
    if abs(d) < 1e-9:
        return f"{body} —"
    improved = (d > 0) if higher_is_better else (d < 0)
    return f"{body} {'✅' if improved else '⚠️'}"


def hms(seconds):
    if seconds is None:
        return "n/a"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def metric_line(label, a, b, is_percent, higher_is_better):
    """One bullet line: label, both values, and the delta.

    Use a bullet so the comparison remains readable in plain-text renderers.
    """
    return (f"- **{label}**: {fmt(a, is_percent)} → {fmt(b, is_percent)}"
            f"  ({fmt_delta(a, b, is_percent, higher_is_better)})")


def higher_is_better(key):
    return key not in ("distance_to_goal", "steps_taken", "path_length")


def build_report(r1, r2, d1, d2, title):
    L = []
    L.append(f"## {title}")
    L.append("")
    L.append(f"- **Run A** `{r1['exp_name']}` — completed {r1['n_episodes']} episodes"
             f" ({d1.get('nprocesses') or '?'} workers, elapsed {hms(d1.get('elapsed_s'))})")
    L.append(f"  - Model: `{d1.get('model_name') or '?'}`")
    L.append(f"- **Run B** `{r2['exp_name']}` — completed {r2['n_episodes']} episodes"
             f" ({d2.get('nprocesses') or '?'} workers, elapsed {hms(d2.get('elapsed_s'))})")
    L.append(f"  - Model: `{d2.get('model_name') or '?'}`")
    L.append("")

    # Anything that differs between the rounds besides the thing under test makes the
    # delta un-attributable. Lead with that rather than burying it.
    confounds = []
    if d1.get('model_name') != d2.get('model_name'):
        confounds.append("**Different models**: metric differences cannot be attributed solely to prompt changes")
    if d1.get('truncated') or d2.get('truncated'):
        which = "Run A" if d1.get('truncated') else "Run B"
        confounds.append(f"**{which} was truncated by the runtime limit**; its episode set is incomplete")
    if r1["n_episodes"] != r2["n_episodes"]:
        confounds.append(f"Different completed episode counts ({r1['n_episodes']} vs {r2['n_episodes']}), "
                         f"Overall metrics use different sets; prioritize the common-episode comparison below")
    if confounds:
        L.append("### ⚠️ Comparison caveats")
        L.append("")
        for c in confounds:
            L.append(f"- {c}")
        L.append("")

    common = set(r1["episode_ids"]) & set(r2["episode_ids"])
    L.append("### Overall metrics (all completed episodes per run)")
    L.append("")
    for key, label, is_pct in METRICS:
        L.append(metric_line(label, r1["means"][key], r2["means"][key],
                             is_pct, higher_is_better(key)))
    L.append("")

    # Same-episode comparison: the fairest read when the two rounds differ in
    # which episodes finished.
    if common and (r1["n_episodes"] != r2["n_episodes"]):
        L.append(f"### Common-episode comparison ({len(common)} episodes)")
        L.append("")
        for key, label, is_pct in METRICS:
            m1, _ = mean_finite(r1["episodes"][e].get(key) for e in common)
            m2, _ = mean_finite(r2["episodes"][e].get(key) for e in common)
            L.append(metric_line(label, m1, m2, is_pct, higher_is_better(key)))
        L.append("")

    if common:
        # Episodes whose success flipped between rounds — the most useful
        # starting point for eyeballing what the prompt change did.
        gained = sorted((e for e in common
                         if not r1["episodes"][e].get("success")
                         and r2["episodes"][e].get("success")),
                        key=lambda x: int(x) if str(x).isdigit() else 0)
        lost = sorted((e for e in common
                       if r1["episodes"][e].get("success")
                       and not r2["episodes"][e].get("success")),
                      key=lambda x: int(x) if str(x).isdigit() else 0)
        L.append("### Episodes with changed success outcomes")
        L.append("")
        L.append(f"- New successes (A failed → B succeeded): {len(gained)}"
                 + (f" — {', '.join(gained)}" if gained else ""))
        L.append(f"- New failures (A succeeded → B failed): {len(lost)}"
                 + (f" — {', '.join(lost)}" if lost else ""))
        L.append("")

    # Compare the distribution of unsuccessful episodes across failure buckets.
    b1, b2 = r1.get("buckets") or {}, r2.get("buckets") or {}
    if b1 or b2:
        L.append("### Failure categories T1–T6")
        L.append("")
        labels = {
            "T1": "Budget exhausted, oracle success, final distance <= 1 m",
            "T2": "Budget exhausted, oracle success, final distance > 1 m",
            "T3": "Budget exhausted, no oracle success",
            "T4": "Below budget threshold, oracle success",
            "T5": "Below budget threshold, no oracle success, final distance <= 2 m",
            "T6": "Below budget threshold, no oracle success, final distance > 2 m",
        }
        for key in ("T1", "T2", "T3", "T4", "T5", "T6"):
            n1, n2 = len(b1.get(key, [])), len(b2.get(key, []))
            delta = n2 - n1
            mark = "—" if delta == 0 else ("✅" if delta < 0 else "⚠️")
            L.append(f"- **{key}** {labels[key]}: {n1} → {n2} ({delta:+d} {mark})")
        if b2.get("T1"):
            L.append(f"  - ⚠️ Run B episodes in T1: {', '.join(b2['T1'])}")
        L.append("")

    # Denominator caveats: non-finite distance_to_goal (navmesh islands) are
    # excluded from the means above, so say how many were dropped.
    nf1, nf2 = r1.get("non_finite") or [], r2.get("non_finite") or []
    if nf1 or nf2:
        L.append(f"- **Non-finite distance (excluded from means)**: "
                 f"Run A {len(nf1)} → Run B {len(nf2)}"
                 + (f" (Run B: {', '.join(nf2)})" if nf2 else ""))
        L.append("")

    L.append("### Runtime health")
    L.append("")
    u1, u2 = r1.get("usage"), r2.get("usage")
    L.append(f"- **API errors**: {d1.get('api_errors', 'n/a')} → {d2.get('api_errors', 'n/a')}")
    L.append(f"- **Fallback decisions**: {d1.get('degraded', 'n/a')} → {d2.get('degraded', 'n/a')}")
    L.append(f"- **Retries**: {d1.get('retries', 'n/a')} → {d2.get('retries', 'n/a')}")
    L.append(f"- **LLM calls**: {u1['calls'] if u1 else 'n/a'} → "
             f"{u2['calls'] if u2 else 'n/a'}")
    L.append(f"- **LLM total tokens**: {u1['total_tokens'] if u1 else 'n/a'} → "
             f"{u2['total_tokens'] if u2 else 'n/a'}")
    h1, h2 = d1.get('harness') or {}, d2.get('harness') or {}
    if any(v for k, v in h1.items() if not k.startswith('_')) or \
       any(v for k, v in h2.items() if not k.startswith('_')):
        L.append("")
        L.append("### Harness health")
        L.append("")
        for key, label in [
            ('HARNESS_EPISODE_START', 'episode starts'),
            ('HARNESS_TOOL_CALL', 'tool calls'),
            ('HARNESS_TOOL_RESULT', 'tool results'),
            ('HARNESS_RECEIPT', 'receipts'),
            ('HARNESS_TERMINAL', 'terminal records'),
            ('HARNESS_VIOLATION', 'violations'),
            ('HARNESS_NOTE', 'Report-only notes (excluded from violations)'),
        ]:
            L.append(f"- **{label}**: {h1.get(key, 0)} → {h2.get(key, 0)}")
        kinds = h2.get('_note_kinds') or {}
        if kinds:
            L.append(f"- **Note/violation counts by kind (Run B)**: "
                     f"{dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}")
            L.append("  - `HARNESS_NOTE:todo_evidence_stale` = TODO completions flagged for "
                     "stale evidence or same-step completion; diagnostic only")
        L.append("")
        L.append("### Place memory")
        L.append("")
        for key, label in [
            ('WORLD_GRAPH_PLACE', 'Place records'),
            ('WORLD_GRAPH_OBJECT', 'Object records'),
            ('WORLD_GRAPH_FAILURE', 'Failure records'),
            ('MEMORY_READOUT', 'Memory readouts supplied to prompts'),
        ]:
            L.append(f"- **{label}**: {h1.get(key, 0)} → {h2.get(key, 0)}")
        L.append("")
        L.append("### STOP event counts")
        L.append("")
        for key, label in [
            ('AUTOSTOP', 'Budget autostop accepted'),
            ('AUTOSTOP_SKIPPED', 'Budget autostop skipped'),
            ('AUTOSTOP_UNAFFORDABLE', 'Insufficient budget for confirmation panorama'),
            ('STOP_GATE_PASS', 'Geometric gate passed'),
            ('STOP_GATE_FAIL', 'Geometric gate failed'),
            ('STOP_REJECTED', 'STOP rejected (total)'),
            ('STOP_FORCED', 'Forced STOP (legacy)'),
        ]:
            L.append(f"- **{label}**: {h1.get(key, 0)} → {h2.get(key, 0)}")
        basis2 = h2.get('_autostop_basis') or {}
        if basis2:
            L.append(f"- **Autostop distance bases (Run B)**: "
                     f"{dict(sorted(basis2.items(), key=lambda kv: -kv[1]))}")
            L.append("  - `fresh_depth` uses current bounding-box depth; `map_target` uses the stored target projection")
    L.append("")
    L.append("### Data sources")
    L.append("")
    L.append(f"- Run A: `{r1['ckpt_dir']}/`, log `{d1.get('log') or 'n/a'}`")
    L.append(f"- Run B: `{r2['ckpt_dir']}/`, log `{d2.get('log') or 'n/a'}`")
    L.append(f"- Completed episodes per worker (Run A): {r1['per_worker']}")
    L.append(f"- Completed episodes per worker (Run B): {r2['per_worker']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round1")
    ap.add_argument("round2")
    ap.add_argument("--title", default="Navigation evaluation comparison")
    ap.add_argument("--out", default=None, help="also write the Markdown here")
    args = ap.parse_args()

    r1, r2 = load_round(args.round1), load_round(args.round2)
    d1, d2 = log_diagnostics(args.round1), log_diagnostics(args.round2)
    d1['harness'] = harness_diagnostics(args.round1)
    d2['harness'] = harness_diagnostics(args.round2)
    report = build_report(r1, r2, d1, d2, args.title)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
