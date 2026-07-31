"""v2 image generation (Pollinations) + AI verification/retry. cells 19-20."""
import json
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
    generation prompt — so a retry doesn't need a second LLM call.
    Returns (ok, reason, refined_prompt). refined_prompt is "" when ok=True.
    """
    img = Image.open(image_path)
    current_prompt = current_prompt or scene_description
    prompt = f"""
You are a strict QA reviewer for AI-generated story illustration frames.

Scene it should depict:
\"\"\"{scene_description}\"\"\"

Prompt that was sent to the image generator to produce this image:
\"\"\"{current_prompt}\"\"\"

Judge the image on:
1. Cleanliness — no garbled text, no mangled hands/faces, no broken anatomy, no watermark artifacts.
2. Match — does it genuinely depict the subject, setting, and mood described above
3. do not more strict, ignore minor detail error. if most of the things match

If, and only if, it fails ("ok": false), also rewrite the generation prompt so a retry avoids this exact
problem — be more specific/constraining about composition, framing, anatomy, or whatever is needed to fix
the issue, while staying faithful to the original scene. Write it as a standalone, self-contained image
prompt (don't mention verification/failure in it). If it passes, leave "refined_prompt" as an empty string.

Return ONLY JSON: {{"ok": true or false, "reason": "short reason", "refined_prompt": "..."}}
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
        return ok, reason, refined_prompt
    except Exception as err:
        print(f"   ⚠️ verification call failed ({err}) — assuming OK to avoid blocking the pipeline")
        return True, "verification unavailable", ""


def generate_verified_image(description, output_path, max_attempts=None):
    """
    Generate an image and verify it against its scene description, retrying
    on failure. Verification and prompt-refinement happen in ONE Gemini
    call: on failure the reviewer also returns a more guided prompt, which
    is used for the next attempt instead of resending the same prompt.
    """
    max_attempts = max_attempts or config.MAX_IMAGE_ATTEMPTS
    last_path = f"{output_path}.png"
    current_prompt = description
    for attempt in range(1, max_attempts + 1):
        img = generate_image(current_prompt, output_path)
        if img is None:
            print(f"   retrying image generation ({attempt}/{max_attempts})")
            continue

        # Always verify against the ORIGINAL scene description, not the
        # (possibly reworded) generation prompt — that's the ground truth.
        ok, reason, refined_prompt = verify_image(last_path, description, current_prompt)
        if ok:
            return last_path
        print(f"   ✗ verification failed: {reason} — regenerating ({attempt}/{max_attempts})")

        if attempt < max_attempts and refined_prompt:
            print(f"   ✏️ refined prompt: {refined_prompt}")
            current_prompt = refined_prompt

    print("   ⚠️ giving up after max attempts — keeping the last generated image")
    return last_path
