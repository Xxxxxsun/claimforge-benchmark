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
import hashlib
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parent
MODEL_MIN = 512          # service-enforced minimum side
ALIGN = 16
OBJECT_PROMPT_TMPL = (
    "Add a single small realistic {cand} in the {pos} of the image. "
    "Place only the one {cand}; keep everything else unchanged. Preserve "
    "lighting, shadows, texture, camera perspective, and JPEG-like realism. "
    "Do not alter unrelated objects or the background."
)
STAIN_PROMPT_TMPL = (
    "Add a realistic irregular {cand} in the {pos} of the image. "
    "Make the stain look naturally embedded on the existing surface, with "
    "uneven edges, subtle discoloration, and texture matching the surrounding "
    "material. Keep all objects, layout, lighting, perspective, and background "
    "unchanged. Do not add text, extra objects, or unrelated marks."
)
CAT_PROMPT_TMPL = (
    "Add a single small but clearly visible cat entirely within the {pos} of the "
    "image. Make the cat look native to the source image: match the surrounding "
    "visual style, resolution, level of detail, sharpness or blur, noise and "
    "compression artifacts, color, lighting, perspective, and depth of field. "
    "Do not make the cat cleaner, sharper, or more photorealistic than the rest "
    "of the image. Show the cat {pose}, {orientation}. Keep its head and eyes "
    "directed into the scene, never toward the viewer or camera. This must feel "
    "like an unposed candid moment, not a front-facing pet portrait. Keep the "
    "whole cat in frame and "
    "confined to that local area; it should occupy roughly one quarter of the "
    "image without dominating the scene. Keep everything else unchanged. Do not "
    "add text, additional cats, or unrelated objects."
)
CAT_POSES = (
    "walking naturally through the scene",
    "standing casually while observing something nearby",
    "sitting in a relaxed, unposed way",
    "crouching as if inspecting or sniffing a nearby surface",
    "resting comfortably",
    "stretching or turning naturally",
)
CAT_ORIENTATIONS = (
    "seen in side profile and facing toward the left side of the scene",
    "seen in side profile and facing toward the right side of the scene",
    "seen from a three-quarter angle and facing away from the viewer",
    "with its back mostly toward the viewer and attention on the surrounding scene",
)
TRASH_CAN_PROMPT_TMPL = (
    "Add exactly one small, ordinary trash can {pos}. Choose a conventional, "
    "unobtrusive floor or ground spot for this exact scene, near a wall or cabinet "
    "when available and clear of walkways, doors, seating, and work areas. Show "
    "the entire bin unobstructed: full rim or lid, both side contours, complete "
    "base, a clear background gap around its whole silhouette, and ample distance "
    "from every image edge. Use modest scene-appropriate scale and perspective "
    "with a subtle contact shadow. Match the source visual style, lighting, color, "
    "depth of field, sharpness or blur, noise, and compression; never make the bin "
    "cleaner, sharper, or more realistic than the source. Keep everything else "
    "unchanged. No furniture-top placement, extra bins, text, logos, crop, or zoom."
)
TRASH_CAN_FLEXIBLE_PROMPT_TMPL = (
    "Add exactly one NEW small, ordinary trash can. Treat the requested location "
    "only as a loose guide: place it {pos}. If that location is blocked, occupied, "
    "unsupported, or too close to an image edge, ignore it and choose the nearest "
    "physically sensible, unobtrusive spot anywhere inside the crop. Prefer open "
    "floor or ground. In a bathroom, use dry tiled floor beside a toilet or vanity. "
    "If no floor or ground is visible anywhere, use a compact tabletop wastebasket "
    "resting fully on a clearly horizontal hard counter, table, or shelf. Never put "
    "it on a bed, pillow, sofa, chair, toilet, toilet tank, bathtub, shower area, "
    "wall, or vertical cabinet face, and never leave it floating. Show the entire "
    "bin unobstructed: full rim or lid, both side contours, complete base, visible "
    "support directly below the base, and a clear gap around its whole silhouette. "
    "Keep ample margin from every image edge; the base must not touch the bottom "
    "frame. Do not overlap, erase, move, or deform any existing person, furniture, "
    "container, or other object. Use a modest scene-appropriate scale, perspective, "
    "and subtle contact shadow. Match the source visual style, lighting, color, "
    "depth of field, resolution, detail, sharpness or blur, noise, and compression; "
    "never make the bin cleaner, sharper, or more realistic than the source. Keep "
    "the framing and everything else unchanged. No crop, zoom, text, logos, loose "
    "trash, additional new bins, or unrelated objects."
)
TRASH_CAN_GLOBAL_PROMPT_TMPL = (
    "Add exactly one clearly visible NEW small, ordinary trash can at the most "
    "physically sensible and unobtrusive location anywhere in this image. The new "
    "bin is the required edit: do not omit it, duplicate it, or turn it into a cup, "
    "pot, bucket, or existing piece of furniture. Prefer open floor or ground near "
    "a wall, cabinet, desk, or nightstand while keeping doors, walkways, seating, "
    "people, dining places, food displays, and work areas clear. In a bathroom, use "
    "dry tiled floor beside a toilet or vanity, never the toilet, tank, bathtub, "
    "shower, or door track. In a bedroom, use visible floor beside furniture, never "
    "a bed, pillow, sofa, or chair. Only if the entire image has no visible floor or "
    "ground, use a compact tabletop wastebasket fully resting on an empty, clearly "
    "horizontal hard shelf, counter, or desk that is not being used for dining or "
    "food preparation. Show the whole bin in front of surrounding objects: full rim "
    "or lid, both side contours, complete body and base, visible support immediately "
    "below the base, and a clear background gap around the entire silhouette. Keep "
    "it well inside the frame with generous margin on every side; its base must not "
    "touch the bottom edge. Do not hide it behind furniture or people. Use modest "
    "scene-appropriate scale and perspective with a subtle contact shadow. Match the "
    "source visual style, lighting, color, depth of field, resolution, detail, "
    "sharpness or blur, noise, and compression; never make the bin or background "
    "cleaner, sharper, or more realistic than the source. Preserve every existing "
    "person, object, surface, and the original framing. No crop, zoom, text, logos, "
    "loose trash, additional new bins, or unrelated changes."
)


def position_phrase(box, size, low=1 / 3, high=2 / 3):
    """Coarse 3x3-grid description of the orange box centre within the crop,
    used to tell a model that has no native region input where to place it."""
    w, h = size
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    col = "left" if cx < low else ("right" if cx > high else "center")
    row = "top" if cy < low else ("bottom" if cy > high else "middle")
    if row == "middle" and col == "center":
        return "center"
    return f"{row}-{col} area"


def trash_can_position_phrase(box, size):
    """Give the trash can a small, center-anchored envelope inside the edit box.

    Coarse direction words such as ``bottom-right`` and bottom-edge coordinates
    encouraged the model to put a large bin on a crop boundary.  Every exported
    trash-can edit region can contain a 15%-wide by 25%-high silhouette when its
    centre is clamped to this safe central range.
    """
    w, h = size
    center_x = min(0.65, max(0.35, (box[0] + box[2]) / 2 / w))
    center_y = min(0.62, max(0.40, (box[1] + box[3]) / 2 / h))
    return (
        "in safe interior space near the requested area (rough guide: center near "
        f"{round(center_x * 100)}% from the left and "
        f"{round(center_y * 100)}% from the top). Use the nearest physically "
        "sensible spot and keep it below about 15% of the crop width and 25% of "
        "its height"
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


def call_edit_legacy(url, model, img, prompt, width, height, steps, seed,
                     timeout=900):
    """Call the original Tencent vLLM fork's custom chat endpoint."""
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


def call_edit_omni(url, model, img, prompt, width, height, steps, seed,
                   bot_task="think", sys_type="en_unified",
                   guidance_scale=None, timeout=900):
    """Call vLLM-Omni's OpenAI-compatible ``/v1/images/edits`` API."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    form = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
        "output_format": "png",
        "num_inference_steps": str(steps),
        "seed": str(seed),
    }
    if bot_task:
        form["bot_task"] = bot_task
    if sys_type:
        form["sys_type"] = sys_type
    if guidance_scale is not None:
        form["guidance_scale"] = str(guidance_scale)

    sess = requests.Session()
    sess.trust_env = False
    resp = sess.post(
        url,
        data=form,
        files={"image": ("input.png", buf.getvalue(), "image/png")},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    images = data.get("data") or []
    if not images or not images[0].get("b64_json"):
        raise RuntimeError(f"no image in response: {json.dumps(data)[:300]}")
    b64 = images[0]["b64_json"]
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def call_edit(args, img, prompt, width, height, seed):
    if args.api_style == "legacy":
        return call_edit_legacy(
            args.url, args.model, img, prompt, width, height,
            args.steps, seed, args.timeout,
        )
    return call_edit_omni(
        args.url, args.model, img, prompt, width, height,
        args.steps, seed, args.bot_task, args.sys_type,
        args.guidance_scale, args.timeout,
    )


def make_prompt(task, position, prompt_kind):
    if task.get("prompt_override"):
        return str(task["prompt_override"])
    if prompt_kind == "cat":
        digest = hashlib.sha256(task["task_id"].encode("utf-8")).digest()
        return CAT_PROMPT_TMPL.format(
            pos=position,
            pose=CAT_POSES[digest[8] % len(CAT_POSES)],
            orientation=CAT_ORIENTATIONS[digest[9] % len(CAT_ORIENTATIONS)],
        )
    elif prompt_kind == "trash-can-flexible":
        tmpl = TRASH_CAN_FLEXIBLE_PROMPT_TMPL
    elif prompt_kind == "trash-can-global":
        tmpl = TRASH_CAN_GLOBAL_PROMPT_TMPL
    elif prompt_kind == "trash-can":
        tmpl = TRASH_CAN_PROMPT_TMPL
    elif prompt_kind == "stain":
        tmpl = STAIN_PROMPT_TMPL
    else:
        tmpl = OBJECT_PROMPT_TMPL
    return tmpl.format(cand=task["candidates"], pos=position)


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
    box = [int(v) for v in task["edit_region_in_context_xyxy"]]
    if args.prompt_kind in {
        "trash-can",
        "trash-can-flexible",
        "trash-can-global",
    }:
        pos = trash_can_position_phrase(box, (W, H))
    else:
        pos = position_phrase(box, (W, H))
    prompt = make_prompt(task, pos, args.prompt_kind)
    # Python's built-in hash is randomized for every process, which makes a
    # resumed batch produce different images. Derive a stable per-task seed.
    seed_key = task["task_id"]
    if args.seed_salt:
        seed_key += f"\0{args.seed_salt}"
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    seed = (int.from_bytes(digest[:8], "big") % 9_000_000) + 1

    edited_up = call_edit(args, up, prompt, tw, th, seed)
    edited = edited_up.resize((W, H), Image.LANCZOS)

    # Default: the saved crop IS the model's full edited blue-box crop, which is
    # what the downstream pipeline splices back. The orange box is only a
    # positional hint (encoded in the prompt above), not a compositing mask.
    # --paste-back optionally reverts everything outside the orange region to
    # the source pixels (maximally pixel-preserving, but adds a splice seam).
    if args.paste_back:
        mask = feathered_mask((W, H), box, args.feather)
        out = Image.composite(edited, crop, mask)
    else:
        out = edited

    return out, prompt, seed, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/images/edits")
    ap.add_argument("--model", default="vllm_hunyuan_image3")
    ap.add_argument("--api-style", choices=["omni", "legacy"], default="omni",
                    help="vLLM-Omni image edit API or the older custom chat API")
    ap.add_argument("--bot-task", default="think_recaption",
                    choices=["think", "recaption", "think_recaption", "vanilla"],
                    help=(
                        "Hunyuan Instruct prompt mode used by vLLM-Omni; "
                        "default matches the checkpoint generation config"
                    ))
    ap.add_argument("--sys-type", default="en_unified",
                    help="Hunyuan system prompt type; pass an empty string to omit")
    ap.add_argument("--guidance-scale", type=float, default=None,
                    help="optional override; Distil deployment defaults to 2.5")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--model-name", default="hunyuan_image3",
                    help="output dir name under generated_crops/")
    ap.add_argument("--tasks", default="annotations/generation_tasks.jsonl")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument(
        "--seed-salt",
        default="",
        help="optional deterministic salt for retrying selected tasks with a new seed",
    )
    ap.add_argument(
        "--prompt-kind",
        choices=[
            "object",
            "stain",
            "cat",
            "trash-can",
            "trash-can-flexible",
            "trash-can-global",
        ],
        default="object",
        help=(
            "prompt family to use; trash-can-flexible may move away from an "
            "unsupported requested location while preserving the complete bin"
        ),
    )
    ap.add_argument("--feather", type=float, default=2.0)
    ap.add_argument("--paste-back", action="store_true",
                    help="revert pixels outside the orange region to source "
                         "(maximally pixel-preserving; adds a splice seam). "
                         "Default: save the model's full edited blue crop.")
    ap.add_argument("--only", default=None,
                    help="comma-separated task_ids or 0-based indices to run")
    ap.add_argument("--resume", action="store_true",
                    help="skip tasks already marked ok whose output files exist")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(REPO / args.tasks)]
    if args.only:
        sel = set(args.only.split(","))
        rows = [r for i, r in enumerate(rows)
                if r["task_id"] in sel or str(i) in sel]

    out_dir = REPO / "generated_crops" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.jsonl"

    if args.resume and man_path.exists():
        completed = set()
        for line in man_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            output_crop = row.get("output_crop")
            if (row.get("status") == "ok" and output_crop
                    and (REPO / output_crop).is_file()):
                completed.add(row["task_id"])
        before = len(rows)
        rows = [row for row in rows if row["task_id"] not in completed]
        print(f"resume: skipping {before - len(rows)} completed task(s); "
              f"{len(rows)} remaining", flush=True)

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
                "service_model": args.model,
                "api_style": args.api_style,
                "bot_task": args.bot_task,
                "sys_type": args.sys_type,
                "prompt_kind": args.prompt_kind,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "prompt": prompt,
                "seed": seed,
                "seed_salt": args.seed_salt,
                "size": [W, H],
                "paste_back": args.paste_back,
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
                "service_model": args.model,
                "api_style": args.api_style,
                "bot_task": args.bot_task,
                "sys_type": args.sys_type,
                "prompt_kind": args.prompt_kind,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "status": "failed",
                "error": repr(e),
            }) + "\n")
            man.flush()
            print(f"[{i+1}/{len(rows)}] {tid} FAILED: {e!r}", flush=True)

    man.close()
    print(f"done: {ok}/{len(rows)} ok. manifest -> {man_path}")


if __name__ == "__main__":
    main()
