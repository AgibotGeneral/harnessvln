"""Probe bbox coordinate conventions using a marker over a real scene frame.

Synthetic blank-canvas probes can disagree with real navigation images. Compare
the predicted box with the known marker box under pixel and normalized scales.

Usage:
    python tools/probe_bbox_real.py [MODEL] [--frame PNG] [--trials N]

Credentials and model settings default to the package-root .env.
"""
import argparse
import base64
import glob
import io
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

W, H = 640, 480                       # the sim's RGB_SENSOR resolution
TRUTH = [200, 150, 420, 330]          # marker box, in absolute pixels

PROMPT = f"""Current situation:
- Image size: {W}x{H} pixels

Your task:
1. Identify the single bright red rectangle overlaid on this scene.
2. Provide its bounding box.

Response format (JSON):
{{
    "reasoning": "<brief explanation>",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description>"
}}"""


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_frame(path):
    """A real scene at sensor resolution, with the marker composited in.

    `combined_step*.png` is rgb and depth side by side, so take the left half.
    """
    img = Image.open(path).convert("RGB")
    if img.width > img.height * 1.5:
        img = img.crop((0, 0, img.width // 2, img.height))
    img = img.resize((W, H))
    ImageDraw.Draw(img).rectangle(
        TRUTH, fill=(220, 30, 30), outline=(120, 0, 0), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def extract_bbox(text):
    m = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text or "")
    if not m:
        return None
    try:
        vals = [float(x.strip()) for x in m.group(1).split(",")]
    except ValueError:
        return None
    return vals[:4] if len(vals) >= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=os.environ.get("MODEL_NAME"))
    ap.add_argument("--key", default=os.environ.get("MODEL_API_KEY"))
    ap.add_argument("--base", default=os.environ.get("MODEL_BASE_URL"))
    ap.add_argument("--frame", default=None, help="a real scene PNG/JPG")
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    if not (a.model and a.key and a.base):
        ap.error("need model, --key and --base (or MODEL_* in .env)")

    frame = a.frame
    if not frame:
        cands = sorted(glob.glob("eval_data/*/rgb/*/combined_step*.png"))
        if not cands:
            ap.error("no --frame given and no eval_data/*/rgb/*/combined_step*.png found")
        frame = cands[0]

    print(f"model : {a.model}")
    print(f"frame : {frame}")
    print(f"truth : {TRUTH}  (absolute pixels in a {W}x{H} frame)")

    client = OpenAI(api_key=a.key, base_url=a.base, timeout=300)
    b64 = build_frame(frame)

    extra = {"enable_thinking": False} if "qwen" in a.model.lower() else None
    px_wins = k_wins = 0
    best_px = best_k = 0.0
    for t in range(a.trials):
        try:
            r = client.chat.completions.create(
                model=a.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                max_tokens=512, temperature=0.0, extra_body=extra)
        except Exception as e:
            print(f"trial {t}: FAILED {type(e).__name__}: {str(e)[:300]}")
            continue
        box = extract_bbox(r.choices[0].message.content)
        if box is None:
            print(f"trial {t}: no bbox_2d parsed")
            continue
        scaled = [box[0] / 1000 * W, box[1] / 1000 * H,
                  box[2] / 1000 * W, box[3] / 1000 * H]
        as_px, as_1000 = iou(box, TRUTH), iou(scaled, TRUTH)
        best_px, best_k = max(best_px, as_px), max(best_k, as_1000)
        px_wins += as_px > as_1000
        k_wins += as_1000 > as_px
        print(f"trial {t}: bbox={[round(v) for v in box]}  "
              f"IoU_pixels={as_px:.3f}  IoU_0-1000={as_1000:.3f} "
              f"-> {[round(v) for v in scaled]}")

    print("\n" + "=" * 60)
    if max(best_px, best_k) < 0.5:
        print(f"VERDICT: neither reading reconstructs the box (best IoU "
              f"{max(best_px, best_k):.3f}). Do NOT run a round on this model.")
        return 1
    if px_wins > k_wins:
        print(f"VERDICT: ABSOLUTE PIXELS ({px_wins}/{px_wins + k_wins} trials, "
              f"best IoU {best_px:.3f} vs {best_k:.3f})")
        print(f"  -> add '{a.model}' to _MODEL_BBOX_SCALE_PIXEL in api.py")
    else:
        print(f"VERDICT: 0-1000 GRID ({k_wins}/{px_wins + k_wins} trials, "
              f"best IoU {best_k:.3f} vs {best_px:.3f})")
        print(f"  -> add '{a.model}' to _MODEL_BBOX_SCALE_0_1000 in api.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
