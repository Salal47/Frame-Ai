"""Chunking, script writing, scene descriptions, TTS narration, audio duration.
Logic unchanged from the v3.3 notebook (cells 13-17); only two things adapted
for headless GitHub Actions runs:
  - `display(Audio(...))` (Colab-only widget) is dropped, replaced with a print.
  - imports come from this package instead of notebook globals.
"""
import os
import json
import time
import wave
import re

from google.genai import types
from langdetect import detect
from pydub import AudioSegment

from .gemini_manager import gemini_text, key_manager


def split_text_for_narration(text, minutes, wpm=150, output_file="chunks.json"):
    """Split `text` into narration-sized chunks (never mid-sentence) via Gemini."""
    target_words = int(minutes * wpm)

    prompt = f"""
Split the following text into narration chunks.

Rules:
- Around {target_words} words per chunk.
- Never split a sentence.
- Preserve the original wording.
- Return ONLY JSON.

Text:
{text}
"""
    response = gemini_text(
        "light",
        prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chunk": {"type": "integer"},
                                "estimated_words": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["chunk", "estimated_words", "text"],
                        },
                    }
                },
                "required": ["chunks"],
            },
        ),
    )

    data = json.loads(response.text)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return os.path.abspath(output_file)


def load_chunks(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)["chunks"]


def create_novel_video_script(text, language="English", style="cinematic storytelling"):
    """The one genuinely *creative* step — uses the heavy model, sparingly."""
    prompt = f"""
Convert this novel text into a short cinematic video narration script.

Language:
{language}

Rules:
- Preserve the original story, characters, and emotions.
- Do not change the plot.
- Make it suitable for YouTube Shorts/TikTok/Reels.
- Start with a strong hook.
- Use natural spoken language.
- Add suspense and curiosity.
- Keep it suitable for voice narration.
- Remove unnecessary descriptions.
- Return ONLY the final script.
- Return ONLY the spoken narration text — no scene labels, no headers,
  no bracketed tags like [Scene 1], no markdown formatting (no **, no #)

Style:
{style}

Novel text:
{text}
"""
    response = gemini_text("creative", prompt)
    return response.text.strip()


def generate_scene_descriptions(script, number_of_scenes):
    """Break the script into visual scene descriptions suitable for AI image generation."""
    number_of_scenes = max(1, round(number_of_scenes))
    prompt = f"""
Convert this cinematic story script into {number_of_scenes} scene descriptions.

Rules:
- Each scene should describe a single visual moment.
- Include characters, location, action, mood, lighting, and camera perspective.
- Keep characters consistent across scenes.
- Make descriptions suitable for AI image generation.
- Do not write narration.
- Return ONLY JSON.

Format:
{{
  "scenes": [
    {{
      "scene": 1,
      "description": "Detailed visual description"
    }}
  ]
}}

Script:
{script}
"""
    response = gemini_text(
        "standard",
        prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)


def clean_script_for_tts(text):
    """Strip scene labels/markdown before sending to TTS — only spoken narration should remain."""
    text = re.sub(r"\*\*\[.*?\]\*\*", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


TTS_VOICE_NAME = "Puck"
TTS_STYLE_INSTRUCTION = (
    "Narrate in an upbeat, casual, youthful MALE voice — like an enthusiastic "
    "teenage boy telling a story to his friends. Do NOT use a girl's or "
    "woman's voice, and do not use a deep, mature adult-male voice."
)
TTS_MAX_ATTEMPTS = 4


def text_to_audio_gemini(text, output_path, max_attempts=None):
    """Narrate `text` with Gemini TTS and save it to `output_path` (.wav)."""
    max_attempts = max_attempts or TTS_MAX_ATTEMPTS
    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "unknown"
    print(f"Detected language: {detected_lang}")
    text = clean_script_for_tts(text)
    styled_text = f"{TTS_STYLE_INSTRUCTION}\n\n{text}"

    def _fn(client, model):
        response = client.models.generate_content(
            model=model,
            contents=styled_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE_NAME)
                    )
                ),
            ),
        )
        candidates = getattr(response, "candidates", None) or []
        content = candidates[0].content if candidates else None
        parts = getattr(content, "parts", None) if content else None
        if not parts or not getattr(parts[0], "inline_data", None):
            finish_reason = getattr(candidates[0], "finish_reason", "unknown") if candidates else "no_candidates"
            raise RuntimeError(f"TTS returned no audio data (finish_reason={finish_reason})")
        return response

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = key_manager.call("tts", _fn)
            audio_data = response.candidates[0].content.parts[0].inline_data.data

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)

            print(f"   🔊 narration saved: {output_path}")
            return output_path
        except Exception as err:
            last_err = err
            print(f"   ⚠️ TTS attempt {attempt}/{max_attempts} failed: {err}")
            if attempt < max_attempts:
                time.sleep(3 * attempt)

    raise RuntimeError(
        f"TTS failed after {max_attempts} attempts (keys/models exhausted or "
        f"repeatedly returned empty audio). Last error: {last_err}"
    )


def get_audio_duration(audio_path):
    """Total audio length in seconds. Supports .mp3 and .wav."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".mp3":
        audio = AudioSegment.from_mp3(audio_path)
    elif ext == ".wav":
        audio = AudioSegment.from_wav(audio_path)
    else:
        raise ValueError("Unsupported format. Use .mp3 or .wav")

    return len(audio) / 1000
