"""One LLM call per chunk that designs the full ordered edit list
(broll / stock / generate segments). cell 34, unchanged logic."""
import json

from google.genai import types

from . import config
from .gemini_manager import gemini_text


def _format_asset_list(assets, kind):
    if not assets:
        return f"(no {kind} clips available)"
    lines = []
    for a in assets:
        lines.append(f'- id={a["id"]} | duration={a["duration"]}s | description="{a["description"]}"')
    return "\n".join(lines)


def plan_video(script, target_duration, broll_options, scene_options,
                broll_ratio=None, language="Urdu"):
    """
    Ask the LLM to design the full edit list for one narration chunk.
    Returns a dict: {"segments": [ {order, source_type, asset_id, prompt,
    duration, reason}, ... ]}
    source_type is one of "broll", "stock", "generate".
    """
    broll_ratio = config.BROLL_RATIO if broll_ratio is None else broll_ratio

    prompt = f"""
You are editing a narrated short-form video ({language} narration). Design the
COMPLETE ordered edit list (list of video segments) for this narration chunk.

TARGET TOTAL DURATION: {target_duration:.2f} seconds. Your segments' durations
must sum to within about 0.3s of this — you have full freedom over the
duration of "generate" segments (no fixed min/max), so use them to make the
total line up exactly. "broll" and "stock" segments have a FIXED, already-
recorded duration (given below) that you cannot change or extend — only
choose WHERE in the sequence to place them and whether the moment fits.

REQUIREMENT: at least {broll_ratio*100:.0f}% of the total duration must be
covered by "broll" segments. If there isn't a good b-roll match for enough of
the timeline, use "stock" or "generate" for the rest, but try hard to hit the
ratio with genuinely well-matched b-roll first.

For every moment, prefer in this order:
1. "broll" — narrator b-roll clip, IF its hand-gesture/body-movement/pace
   description genuinely matches what the narration is saying and its FIXED
   duration fits naturally at that point in the timeline. Lips don't matter
   (masked) but gesture, pacing, and background must fit the moment.
2. "stock" — an existing stock/scene clip, IF its description matches the
   scene/setting/mood better than generating a new image would.
3. "generate" — only if neither library has a genuine match. Give a
   detailed AI-image-generation prompt (subject, setting, action, mood,
   lighting, camera angle) and choose whatever duration makes sense for that
   beat. IMPORTANT: if the scene includes a person, the prompt MUST specify a
   wide or medium shot (e.g. "medium shot", "full-body shot", "wide
   establishing shot") — never a tight close-up/face-focused framing.

Keep characters/setting consistent across "generate" segments.

AVAILABLE B-ROLL CLIPS:
{_format_asset_list(broll_options, 'broll')}

AVAILABLE STOCK/SCENE CLIPS:
{_format_asset_list(scene_options, 'stock')}

NARRATION SCRIPT (read top to bottom, matches the audio timeline):
{script}

Return ONLY JSON in this exact shape:
{{
  "segments": [
    {{
      "order": 1,
      "source_type": "broll" | "stock" | "generate",
      "asset_id": "<id from the lists above, or null if generate>",
      "prompt": "<detailed image prompt, or null if broll/stock>",
      "duration": <seconds, float>,
      "reason": "<why this asset/moment fits here, 1 sentence>"
    }}
  ]
}}
"""
    response = gemini_text(
        "creative", prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text)


def reconcile_plan(plan, target_duration, broll_lookup, scene_lookup,
                    broll_ratio=None, tolerance=None):
    """
    Snap broll/stock segment durations to their REAL clip duration (never
    trust the LLM's number for these), then rescale only the "generate"
    segments proportionally so the total matches target_duration. Also
    reports whether the broll ratio requirement was actually met.
    """
    broll_ratio = config.BROLL_RATIO if broll_ratio is None else broll_ratio
    tolerance = config.PLAN_DURATION_TOLERANCE_SECS if tolerance is None else tolerance

    segs = sorted(plan["segments"], key=lambda s: s["order"])
    fixed_total, gen_total = 0.0, 0.0
    for s in segs:
        if s["source_type"] in ("broll", "stock"):
            lookup = broll_lookup if s["source_type"] == "broll" else scene_lookup
            asset = lookup.get(s["asset_id"])
            if asset is None:
                print(f'   ⚠️ plan referenced unknown asset_id={s["asset_id"]!r} — demoting to generate')
                s["source_type"], s["asset_id"] = "generate", None
                s["prompt"] = s.get("prompt") or s.get("reason") or "cinematic scene matching the narration"
                gen_total += max(1.0, float(s.get("duration", 3)))
                continue
            s["duration"] = asset["duration"]
            fixed_total += s["duration"]
        else:
            gen_total += max(0.5, float(s.get("duration", 3)))

    broll_secs = sum(s["duration"] for s in segs if s["source_type"] == "broll")
    total = fixed_total + gen_total
    delta = target_duration - total

    if abs(delta) > tolerance and gen_total > 0:
        scale = max(0.1, (gen_total + delta) / gen_total)
        for s in segs:
            if s["source_type"] == "generate":
                s["duration"] = round(max(0.5, float(s["duration"])) * scale, 2)

    achieved_ratio = broll_secs / target_duration if target_duration > 0 else 0
    if achieved_ratio + 1e-6 < broll_ratio:
        print(f"   ⚠️ b-roll ratio target {broll_ratio:.0%} not met (got {achieved_ratio:.0%}) — "
              f"not enough matching b-roll was available/appropriate for this chunk")

    plan["segments"] = segs
    return plan
