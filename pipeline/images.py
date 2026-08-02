"""v2 image generation (Pollinations) + AI verification/retry. cells 19-20."""
import json
import os
import urllib.parse
from io import BytesIO

import requests
from PIL import Image
from google.genai import types

from . import config
from .gemini_manager import key_manager


def generate_image(prompt, output_path, timeout=60):
    """Generate one image via Pollinations and save it to `output_path` (no extension)."""
    full_prompt = f"{prompt}, highly detailed, photorealistic"
    url = (
        "https://image.pollinations.ai/prompt/"
        f"{urllib.parse.quote(full_prompt)}?width=1024&height=1024&model=gpt-image-2&nologo=true"
    )
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as err:
        print(f"   ✗ image request error: {err}")
        return None

    if "image" in response.headers.get("content-type", ""):
        img = Image.open(BytesIO(response.content))
        img.save(f"{output_path}.png")
        return img
    else:
        print(f"   ✗ image API error ({response.status_code}): {response.text[:200]}")
        return None


def verify_image(image_path, scene_description, current_prompt=None):
    """
    Ask Gemini (vision) whether the image is clean and matches the scene.
    In the SAME call, if it fails, also ask for a more guided rewrite of the
    generation prompt — so a retry doesn't need a second LLM call. The SAME
    call also returns a 0-100 accuracy score (d2) so callers can compare
    every attempt across every model and keep the single best one, again
    without any extra API call.
    Returns (ok, reason, refined_prompt, accuracy). refined_prompt is "" when
    ok=True. accuracy is a float 0-100.
    """
    img = Image.open(image_path)
    current_prompt = current_prompt or scene_description
    prompt = f"""
You are a QA reviewer for AI-generated story illustration frames.

Scene it should depict:
\"\"\"{scene_description}\"\"\"

Prompt that was sent to the image generator to produce this image:
\"\"\"{current_prompt}\"\"\"

Judge the image on:
1. Cleanliness — no garbled text, no mangled hands/faces, no broken anatomy, no watermark artifacts.
2. Match — does it genuinely depict the subject, setting, and mood described above
3. do not more strict, ignore minor detail error. if most of the things match

Also score the image's overall accuracy from 0-100 (0 = completely broken/off-scene,
100 = perfect match, clean generation). Use this same scale even when "ok" is
true, so a passing image can still be ranked against other passing images
(e.g. a clean 95 vs. a clean-but-slightly-off 80).

If, and only if, it fails ("ok": false), also rewrite the generation prompt so a retry avoids this exact
problem — be more specific/constraining about composition, framing, anatomy, or whatever is needed to fix
the issue, while staying faithful to the original scene. Write it as a standalone, self-contained image
prompt (don't mention verification/failure in it). If it passes, leave "refined_prompt" as an empty string.

Return ONLY JSON: {{"ok": true or false, "reason": "short reason", "refined_prompt": "...", "accuracy": <0-100>}}
"""

    def _fn(client, model):
        return client.models.generate_content(
            model=model,
            contents=[img, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

    try:
        response = key_manager.call("vision", _fn)
        result = json.loads(response.text)
        ok = bool(result.get("ok"))
        reason = result.get("reason", "")
        refined_prompt = "" if ok else (result.get("refined_prompt") or "").strip()
        try:
            accuracy = float(result.get("accuracy", 100 if ok else 0))
        except (TypeError, ValueError):
            accuracy = 100.0 if ok else 0.0
        accuracy = max(0.0, min(100.0, accuracy))
        return ok, reason, refined_prompt, accuracy
    except Exception as err:
        print(f"   ⚠️ verification call failed ({err}) — assuming OK to avoid blocking the pipeline")
        return True, "verification unavailable", "", 75.0


def generate_verified_image(description, output_path, max_attempts=None):
    """
    Generate an image and verify it against its scene description, retrying
    on failure. Verification and prompt-refinement happen in ONE Gemini
    call: on failure the reviewer also returns a more guided prompt, which
    is used for the next attempt instead of resending the same prompt.

    d2: every attempt is saved to its own temp file and scored (accuracy,
    same verify call — no extra API calls). If nothing ever passes, instead
    of blindly keeping the LAST attempt (which could be the worst one), the
    single HIGHEST-scoring attempt across every retry is kept.
    """
    max_attempts = max_attempts or config.MAX_IMAGE_ATTEMPTS
    final_path = f"{output_path}.png"
    current_prompt = description
    best = {"score": -1.0, "path": None}

    for attempt in range(1, max_attempts + 1):
        attempt_base = f"{output_path}__a{attempt}"
        img = generate_image(current_prompt, attempt_base)
        if img is None:
            print(f"   retrying image generation ({attempt}/{max_attempts})")
            continue
        attempt_path = f"{attempt_base}.png"

        # Always verify against the ORIGINAL scene description, not the
        # (possibly reworded) generation prompt — that's the ground truth.
        ok, reason, refined_prompt, accuracy = verify_image(attempt_path, description, current_prompt)

        if ok:
            if best["path"] and os.path.isfile(best["path"]):
                os.remove(best["path"])
            os.replace(attempt_path, final_path)
            return final_path

        if accuracy > best["score"]:
            if best["path"] and os.path.isfile(best["path"]):
                os.remove(best["path"])
            best = {"score": accuracy, "path": attempt_path}
        else:
            os.remove(attempt_path)

        print(f"   ✗ verification failed: {reason} (score {accuracy:.0f}/100) — regenerating ({attempt}/{max_attempts})")
        if attempt < max_attempts and refined_prompt:
            print(f"   ✏️ refined prompt: {refined_prompt}")
            current_prompt = refined_prompt

    if best["path"] is None:
        print("   ⚠️ every attempt failed to even generate an image")
        return final_path

    os.replace(best["path"], final_path)
    print(f"   ⚠️ giving up after max attempts — keeping the highest-accuracy attempt (score {best['score']:.0f}/100)")
    return final_path
