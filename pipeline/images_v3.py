"""v3 multi-model image generation with automatic rollout across
IMAGE_MODEL_CHAIN + LLM-chosen b-roll/stock fallback when every model fails
verification. cell 36, unchanged logic."""
import base64
import json
import urllib.parse
from io import BytesIO

import requests
from PIL import Image
from google.genai import types

from . import config
from .gemini_manager import gemini_text
from .images import verify_image
from .planner import _format_asset_list


def _gen_image_openai(model_name, prompt, output_path, timeout=90):
    """gpt-image-2 / gptimage via OpenAI's Images API."""
    if not config.OPENAI_API_KEY:
        return None
    w, h = config.IMAGE_GEN_SIZE
    full_prompt = (
        f"{prompt}, highly detailed, photorealistic, wide or medium shot, "
        f"full scene visible — NOT a close-up, NOT a tight face crop"
    )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"model": model_name, "prompt": full_prompt, "size": f"{w}x{h}"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"   ✗ {model_name} error ({resp.status_code}): {resp.text[:200]}")
            return None
        b64 = resp.json()["data"][0]["b64_json"]
        img = Image.open(BytesIO(base64.b64decode(b64)))
        img.save(f"{output_path}.png")
        return img
    except Exception as err:
        print(f"   ✗ {model_name} request error: {err}")
        return None


def _gen_image_pollinations(model_name, prompt, output_path, timeout=60):
    """flux / sana / klein / zimage / gptimage-proxy via Pollinations' model= param."""
    w, h = config.IMAGE_GEN_SIZE
    full_prompt = (
        f"{prompt}, highly detailed, photorealistic, wide or medium shot, "
        f"full scene visible, NOT a close-up, NOT a tight face crop"
    )
    url = (
        "https://image.pollinations.ai/prompt/"
        f"{urllib.parse.quote(full_prompt)}?width={w}&height={h}&model={model_name}&nologo=true"
    )
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as err:
        print(f"   ✗ {model_name} request error: {err}")
        return None
    if "image" in response.headers.get("content-type", ""):
        img = Image.open(BytesIO(response.content))
        img.save(f"{output_path}.png")
        return img
    print(f"   ✗ {model_name} API error ({response.status_code}): {response.text[:200]}")
    return None


def generate_image_with_model(model_name, prompt, output_path):
    if model_name in ("gpt-image-2", "gptimage") and config.OPENAI_API_KEY:
        return _gen_image_openai(model_name, prompt, output_path)
    return _gen_image_pollinations(model_name, prompt, output_path)


def llm_pick_fallback_asset(description, broll_options, scene_options):
    """Called only when every image model failed verification for a scene."""
    prompt = f"""
Image generation for this scene kept coming back broken or off-scene after
trying every available model:
\"\"\"{description}\"\"\"

Pick the SINGLE best-matching clip from the lists below as a substitute.

B-ROLL CLIPS:
{_format_asset_list(broll_options, 'broll')}

STOCK/SCENE CLIPS:
{_format_asset_list(scene_options, 'stock')}

Return ONLY JSON: {{"source_type": "broll" or "stock", "asset_id": "<id>", "reason": "..."}}
"""
    response = gemini_text(
        "standard", prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)


def generate_verified_image_v3(description, output_path, broll_lookup, scene_lookup,
                                max_attempts_per_model=None):
    """
    Try each model in IMAGE_MODEL_CHAIN in order (with retries per model),
    verifying against the scene description each time. If every model fails,
    ask the LLM to substitute a b-roll/stock clip instead.
    """
    max_attempts_per_model = max_attempts_per_model or config.MAX_ATTEMPTS_PER_IMAGE_MODEL
    last_path = f"{output_path}.png"

    for model_name in config.IMAGE_MODEL_CHAIN:
        for attempt in range(1, max_attempts_per_model + 1):
            img = generate_image_with_model(model_name, description, output_path)
            if img is None:
                print(f"   [{model_name}] generation failed, retry {attempt}/{max_attempts_per_model}")
                continue
            ok, reason = verify_image(last_path, description)
            if ok:
                return {"source_type": "generate", "path": last_path, "model": model_name}
            print(f"   [{model_name}] ✗ verification failed: {reason} (attempt {attempt}/{max_attempts_per_model})")
        print(f"   [{model_name}] exhausted — rolling out to next image model")

    print("   ⚠️ all image models failed for this scene — asking the LLM for a b-roll/stock fallback")
    all_broll = list(broll_lookup.values())
    all_scenes = list(scene_lookup.values())
    fallback = llm_pick_fallback_asset(description, all_broll, all_scenes)
    lookup = broll_lookup if fallback.get("source_type") == "broll" else scene_lookup
    asset = lookup.get(fallback.get("asset_id"))
    if asset is None:
        print("   ⚠️ LLM fallback also failed to resolve — keeping the last (broken) generated image")
        return {"source_type": "generate", "path": last_path, "model": "none"}
    return {"source_type": fallback["source_type"], "asset_id": asset["id"], "path": asset["path"]}
