#!/usr/bin/env python3
"""CLAIMFORGE generation agent for the deployed HunyuanImage-3.0 I2I service.

For each task in annotations/generation_tasks.jsonl:
  1. Load the (tiny) context_crop and remember its exact size.
  2. Upscale it so the short side is >= MODEL_MIN px (model requires 512-2048),
     16-aligned, preserving aspect ratio.
  3. Run the deployed I2I edit service (image-conditioned) with the task prompt.
  4. Downscale the model output back to the exact original context_crop size.
  5. Paste the edited region back over the original crop using a feathered mask
     built from edit_region_in_context_xyxy -> only the insert region changes,
     the rest of the crop stays pixel-identical to the input ("pixel-preserving
     local edit").

Outputs: generated_crops/<model>/<task_id>.png + manifest.jsonl
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parent
MODEL_MIN = 512          # service-enforced minimum side
ALIGN = 16
PROMPT_TMPL = (
    "Add a small realistic {cand} inside the marked target area. Keep the rest "
    "of the image unchanged. Preserve lighting, shadows, texture, camera "
    "perspective, and JPEG-like realism. Do not alter unrelated objects or the "
    "background."
)


def ceil_to(x, m=ALIGN):
    return ((int(round(x)) + m - 1) // m) * m


def upscale_size(w, h):
    s = MODEL_MIN / min(w, h)
    return ceil_to(w * s), ceil_to(h * s)


def b64_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def call_edit(url, model, img, prompt, width, height, steps, seed, timeout=900):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 1,
        "temperature": 0,
        "seed": seed,
        "chat_template": (
            "{% for message in messages %}{% if message['role'] == 'user' %}"
            "<|startoftext|>{{ message['content'] }}{% endif %}{% endfor %}"
        ),
        "task_type": "hunyuan_image3",
        "task_extra_kwargs": {
            "diff_infer_steps": steps,
            "use_system_prompt": "None",
            "bot_task": "image",
            "image_size": f"{height}x{width}",
            "image": [b64_data_uri(img)],
        },
    }
    sess = requests.Session()
    sess.trust_env = False
    resp = sess.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    b64 = data.get("image")
    if not b64:
        raise RuntimeError(f"no image in response: {json.dumps(data)[:300]}")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def feathered_mask(size, box, feather):
    """Binary-ish soft mask: white inside `box` (xyxy), blurred edges."""
    w, h = size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return m


def run_task(task, args):
    ctx_path = REPO / task["context_crop"]
    crop = Image.open(ctx_path).convert("RGB")
    W, H = crop.size
    tw, th = upscale_size(W, H)

    up = crop.resize((tw, th), Image.LANCZOS)
    prompt = PROMPT_TMPL.format(cand=task["candidates"])
    seed = (abs(hash(task["task_id"])) % 9_000_000) + 1

    edited_up = call_edit(args.url, args.model, up, prompt, tw, th,
                          args.steps, seed)
    edited = edited_up.resize((W, H), Image.LANCZOS)

    # Paste edited pixels back only inside the (feathered) insert region so the
    # rest of the crop stays identical to the source.
    out = crop.copy()
    if not args.no_paste_back:
        box = [int(v) for v in task["edit_region_in_context_xyxy"]]
        mask = feathered_mask((W, H), box, args.feather)
        out = Image.composite(edited, crop, mask)
    else:
        out = edited

    return out, prompt, seed, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    ap.add_argument("--model", default="vllm_hunyuan_image3")
    ap.add_argument("--model-name", default="hunyuan_image3",
                    help="output dir name under generated_crops/")
    ap.add_argument("--tasks", default="annotations/generation_tasks.jsonl")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--feather", type=float, default=2.0)
    ap.add_argument("--no-paste-back", action="store_true",
                    help="save raw model output (whole crop) instead of masked paste-back")
    ap.add_argument("--only", default=None,
                    help="comma-separated task_ids or 0-based indices to run")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(REPO / args.tasks)]
    if args.only:
        sel = set(args.only.split(","))
        rows = [r for i, r in enumerate(rows)
                if r["task_id"] in sel or str(i) in sel]

    out_dir = REPO / "generated_crops" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.jsonl"
    man = open(man_path, "a")

    ok = 0
    for i, task in enumerate(rows):
        tid = task["task_id"]
        t0 = time.time()
        try:
            out, prompt, seed, (W, H) = run_task(task, args)
            out_path = out_dir / f"{tid}.png"
            out.save(out_path)
            assert out.size == (W, H), f"size mismatch {out.size} != {(W, H)}"
            man.write(json.dumps({
                "task_id": tid,
                "input_context_crop": task["context_crop"],
                "output_crop": str(out_path.relative_to(REPO)),
                "model": args.model_name,
                "prompt": prompt,
                "seed": seed,
                "size": [W, H],
                "paste_back": not args.no_paste_back,
                "status": "ok",
            }) + "\n")
            man.flush()
            ok += 1
            print(f"[{i+1}/{len(rows)}] {tid} {task['candidates']:9s} "
                  f"{W}x{H} {time.time()-t0:.1f}s -> {out_path.name}", flush=True)
        except Exception as e:
            man.write(json.dumps({
                "task_id": tid,
                "input_context_crop": task["context_crop"],
                "model": args.model_name,
                "status": "failed",
                "error": repr(e),
            }) + "\n")
            man.flush()
            print(f"[{i+1}/{len(rows)}] {tid} FAILED: {e!r}", flush=True)

    man.close()
    print(f"done: {ok}/{len(rows)} ok. manifest -> {man_path}")


if __name__ == "__main__":
    main()
