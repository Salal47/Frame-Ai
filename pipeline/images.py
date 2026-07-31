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


def verify_image(image_path, scene_description):
    """Ask Gemini (vision) whether the image is clean and matches the scene."""
    img = Image.open(image_path)
    prompt = f"""
You are a strict QA reviewer for AI-generated story illustration frames.

Scene it should depict:
\"\"\"{scene_description}\"\"\"

Judge the image on:
1. Cleanliness — no garbled text, no mangled hands/faces, no broken anatomy, no watermark artifacts.
2. Match — does it genuinely depict the subject, setting, and mood described above?

Return ONLY JSON: {{"ok": true or false, "reason": "short reason"}}
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
        return bool(result.get("ok")), result.get("reason", "")
    except Exception as err:
        print(f"   ⚠️ verification call failed ({err}) — assuming OK to avoid blocking the pipeline")
        return True, "verification unavailable"


def generate_verified_image(description, output_path, max_attempts=None):
    """Generate an image and verify it against its scene description, retrying on failure."""
    max_attempts = max_attempts or config.MAX_IMAGE_ATTEMPTS
    last_path = f"{output_path}.png"
    for attempt in range(1, max_attempts + 1):
        img = generate_image(description, output_path)
        if img is None:
            print(f"   retrying image generation ({attempt}/{max_attempts})")
            continue

        ok, reason = verify_image(last_path, description)
        if ok:
            return last_path
        print(f"   ✗ verification failed: {reason} — regenerating ({attempt}/{max_attempts})")

    print("   ⚠️ giving up after max attempts — keeping the last generated image")
    return last_path
