"""v3 multi-model image generation with automatic rollout across
IMAGE_MODEL_CHAIN. d2: no more b-roll/stock fallback when every model fails
verification — instead, every attempt from every model is scored and the
single highest-accuracy attempt across the whole chain is kept. Prompts are
also now written per-model (see config.IMAGE_MODEL_PROMPT_HINTS) since not
every model in the chain is equally capable."""
import base64
import os
import urllib.parse
from io import BytesIO

import requests
from PIL import Image

from . import config
from .images import verify_image


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


def _model_aware_prompt(model_name, base_prompt):
    """d2: not every model in IMAGE_MODEL_CHAIN is equally powerful, so the
    generation prompt is written WITH the model name in mind instead of
    sending the exact same prompt to every model. Uses
    config.IMAGE_MODEL_PROMPT_HINTS (falls back to a generic hint for an
    unlisted model)."""
    hint = config.IMAGE_MODEL_PROMPT_HINTS.get(model_name, config.DEFAULT_IMAGE_MODEL_HINT)
    return f"{base_prompt}\n\n[Guidance for the '{model_name}' image model: {hint}]"


def generate_verified_image_v3(description, output_path, max_attempts_per_model=None):
    """
    Try each model in IMAGE_MODEL_CHAIN in order (with retries per model),
    verifying against the scene description each time and scoring every
    attempt (accuracy comes back in the SAME verify_image call — no extra
    API calls). Each attempt is written to its own temp file; whichever
    attempt scores highest is the one kept.

    d2: there is NO b-roll/stock fallback anymore. If nothing ever passes
    verification, the single highest-accuracy attempt seen across every
    model/retry is used instead of silently substituting a b-roll clip.
    """
    max_attempts_per_model = max_attempts_per_model or config.MAX_ATTEMPTS_PER_IMAGE_MODEL
    final_path = f"{output_path}.png"
    best = {"score": -1.0, "path": None, "model": None}

    for model_name in config.IMAGE_MODEL_CHAIN:
        current_prompt = _model_aware_prompt(model_name, description)

        for attempt in range(1, max_attempts_per_model + 1):
            attempt_base = f"{output_path}__{model_name.replace('-', '_')}_a{attempt}"
            img = generate_image_with_model(model_name, current_prompt, attempt_base)
            if img is None:
                print(f"   [{model_name}] generation failed, retry {attempt}/{max_attempts_per_model}")
                continue
            attempt_path = f"{attempt_base}.png"

            # Always verify against the ORIGINAL scene description, not the
            # (possibly reworded) generation prompt — that's the ground truth.
            # verify_image returns a refined prompt + accuracy score in the
            # SAME vision call, so no extra Gemini call is needed here.
            ok, reason, refined_prompt, accuracy = verify_image(attempt_path, description, current_prompt)

            if ok:
                print(f"   [{model_name}] ✔ verified (score {accuracy:.0f}/100)")
                if best["path"] and os.path.isfile(best["path"]):
                    os.remove(best["path"])
                os.replace(attempt_path, final_path)
                return {"source_type": "generate", "path": final_path, "model": model_name, "accuracy": accuracy}

            if accuracy > best["score"]:
                if best["path"] and os.path.isfile(best["path"]):
                    os.remove(best["path"])
                best = {"score": accuracy, "path": attempt_path, "model": model_name}
            else:
                os.remove(attempt_path)

            print(f"   [{model_name}] ✗ verification failed: {reason} "
                  f"(score {accuracy:.0f}/100, attempt {attempt}/{max_attempts_per_model})")
            if refined_prompt:
                print(f"   ✏️ refined prompt: {refined_prompt}")
                current_prompt = _model_aware_prompt(
                    model_name, refined_prompt,
                )
        print(f"   [{model_name}] exhausted — rolling out to next image model")

    if best["path"] is None:
        print("   ⚠️ every image model failed to even generate an image for this scene")
        return {"source_type": "generate", "path": final_path, "model": "none", "accuracy": 0}

    os.replace(best["path"], final_path)
    print(f"   ⚠️ no model passed verification for this scene — keeping the highest-accuracy "
          f"attempt anyway ({best['model']}, score {best['score']:.0f}/100)")
    return {"source_type": "generate", "path": final_path, "model": best["model"], "accuracy": best["score"]}
