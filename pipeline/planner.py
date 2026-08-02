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


def _normalize_plan(data):
    """
    Gemini sometimes returns a bare JSON array (`[...]`) instead of the
    requested `{"segments": [...]}` wrapper, even with response_schema-less
    response_mime_type="application/json" — that's valid JSON, just not the
    shape we asked for. Normalize it here so downstream code (reconcile_plan
    etc.) can always assume `plan["segments"]` exists, instead of crashing
    with something like `list indices must be integers or slices, not str`.
    """
    if isinstance(data, list):
        return {"segments": data}
    if isinstance(data, dict) and "segments" not in data:
        # Some other wrapper key (e.g. {"plan": [...]}) — use the first
        # list-of-dicts value found.
        for value in data.values():
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                return {"segments": value}
    return data


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

TIMING MUST BE TIGHT: each segment's visual has to be on screen AT OR BEFORE
the moment the narration talks about it — never after. Don't let the
narration finish describing something before its matching visual appears.
Since "broll"/"stock" have a fixed duration but "generate" segments don't,
use the free-form length of "generate" segments to pull visuals earlier or
stretch/shrink them so the picture and the words line up tightly — you are
free to make a "generate" segment shorter or longer than neighboring beats
purely to keep this sync correct.

HARD LIMIT: "broll" segments must cover AT MOST {broll_ratio*100:.0f}% of the
total duration — never more. Only reach for "broll" when its hand-gesture/
body-movement/pace description genuinely matches the narration AND you still
have headroom under that {broll_ratio*100:.0f}% ceiling; once you're at or near
it, use "stock" or "generate" for everything else even if another b-roll clip
would technically fit.

For every moment, prefer in this order (subject to the {broll_ratio*100:.0f}%
b-roll ceiling above):
1. "broll" — narrator b-roll clip, IF its hand-gesture/body-movement/pace
   description genuinely matches what the narration is saying, its FIXED
   duration fits naturally at that point in the timeline, AND using it won't
   push the b-roll total over the {broll_ratio*100:.0f}% ceiling. Lips don't
   matter (masked) but gesture, pacing, and background must fit the moment.
2. "stock" — an existing stock/scene clip, IF its description matches the
   scene/setting/mood better than generating a new image would.
3. "generate" — the default whenever neither library has a genuine match, or
   the b-roll ceiling is already used up, or tight narration-sync needs a
   duration a fixed clip can't provide. Give a detailed AI-image-generation
   prompt (subject, setting, action, mood, lighting, camera angle) and choose
   whatever duration keeps the visual synced to the narration at that beat.
   IMPORTANT: if the scene includes a person, the prompt MUST specify a wide
   or medium shot (e.g. "medium shot", "full-body shot", "wide establishing
   shot") — never a tight close-up/face-focused framing.

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
    return _normalize_plan(json.loads(response.text))


def reconcile_plan(plan, target_duration, broll_lookup, scene_lookup,
                    broll_ratio=None, tolerance=None):
    """
    Snap broll/stock segment durations to their REAL clip duration (never
    trust the LLM's number for these), then rescale only the "generate"
    segments proportionally so the total matches target_duration.

    d2: broll_ratio is now a hard MAX CEILING, not a minimum floor. This is
    enforced here as a guarantee (not just a prompt instruction) — if the
    LLM's plan put more than broll_ratio's worth of b-roll on the timeline,
    the excess "broll" segments are demoted to "generate" (largest first, so
    the fewest segments need demoting) until the ceiling is satisfied.
    """
    broll_ratio = config.BROLL_RATIO if broll_ratio is None else broll_ratio
    tolerance = config.PLAN_DURATION_TOLERANCE_SECS if tolerance is None else tolerance

    plan = _normalize_plan(plan)
    if not isinstance(plan, dict) or not isinstance(plan.get("segments"), list):
        raise ValueError(
            f"plan_video() returned a shape reconcile_plan can't use "
            f"(expected a dict with a 'segments' list): {plan!r}"
        )

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

    # --- HARD CEILING: b-roll must never exceed broll_ratio of the total ---
    broll_secs = sum(s["duration"] for s in segs if s["source_type"] == "broll")
    max_broll_secs = broll_ratio * target_duration
    if broll_secs > max_broll_secs + 1e-6:
        excess = broll_secs - max_broll_secs
        broll_segs_desc = sorted(
            (s for s in segs if s["source_type"] == "broll"),
            key=lambda s: s["duration"], reverse=True,
        )
        demoted = 0
        for s in broll_segs_desc:
            if excess <= 1e-6:
                break
            reclaimed = s["duration"]
            s["source_type"], s["asset_id"] = "generate", None
            s["prompt"] = s.get("prompt") or s.get("reason") or "cinematic scene matching the narration"
            s["duration"] = max(1.0, reclaimed)
            fixed_total -= reclaimed
            gen_total += s["duration"]
            broll_secs -= reclaimed
            excess -= reclaimed
            demoted += 1
        print(f"   ⚠️ b-roll ceiling ({broll_ratio:.0%}) exceeded — demoted {demoted} "
              f"broll segment(s) to 'generate' to stay under the cap")

    total = fixed_total + gen_total
    delta = target_duration - total

    if abs(delta) > tolerance and gen_total > 0:
        scale = max(0.1, (gen_total + delta) / gen_total)
        for s in segs:
            if s["source_type"] == "generate":
                s["duration"] = round(max(0.5, float(s["duration"])) * scale, 2)

    achieved_ratio = broll_secs / target_duration if target_duration > 0 else 0
    print(f"   ℹ️ b-roll coverage: {achieved_ratio:.0%} of this chunk (ceiling {broll_ratio:.0%})")

    plan["segments"] = segs
    return plan
