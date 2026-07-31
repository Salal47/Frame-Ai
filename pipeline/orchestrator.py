"""
Stepped version of the notebook's process_and_generate_v3 (cell 40).

Same resumable-on-Drive contract (chunks.json, run_state.json, per-chunk
script/narration/plan/images, chunk videos) — the only behavioural change is
that `process_file_step()` processes at most `max_new_chunks` chunks that
weren't already done, then returns control to the caller instead of looping
over every remaining chunk. That's what lets main.py switch between 2+
source files every 2-3 chunks instead of draining one file per run.

If Gemini exhausts every fallback model for a task mid-chunk,
gemini_manager.AllModelsExhaustedError propagates straight up through this
function uncaught — main.py catches it once, at the top level, and stops the
whole run (partial progress is already saved in run_state.json, so the next
hourly run resumes exactly here).
"""
import os
from pathlib import Path

from . import config
from .content_gen import (
    split_text_for_narration, load_chunks, create_novel_video_script,
    text_to_audio_gemini, get_audio_duration,
)
from .drive_sync import wait_for_upload
from .images_v3 import generate_verified_image_v3
from .library import load_libraries
from .planner import plan_video, reconcile_plan
from .run_state import load_run_state, save_run_state
from .source_utils import load_text
from .video_assembly import assemble_final_video
from .video_assembly_v3 import assemble_chunk_video_v3


def _run_dir_for(file_path, output_dir):
    video_name = Path(file_path).stem
    return video_name, os.path.join(output_dir, video_name + "_v3")


def process_file_step(file_path, max_new_chunks, output_dir=None, language="Urdu",
                       minutes_per_chunk=3, broll_ratio=None, libraries=None):
    """
    Process up to `max_new_chunks` not-yet-done chunks of `file_path`, then
    return. Chunks that were already finished in a previous run are skipped
    for free and don't count against the budget.

    Returns a dict:
      {"status": "complete", "final_path": ..., "chunks_done_this_call": N}
      {"status": "partial",  "chunks_done_this_call": N, "total_chunks": M, "done_chunks": K}
    """
    output_dir = output_dir or config.OUTPUT_ROOT
    broll_ratio = config.BROLL_RATIO if broll_ratio is None else broll_ratio

    text = load_text(file_path)
    video_name, run_dir = _run_dir_for(file_path, output_dir)
    images_dir = os.path.join(run_dir, "images")
    audio_dir = os.path.join(run_dir, "audio")
    chunks_dir = os.path.join(run_dir, "chunk_videos")
    for d in (images_dir, audio_dir, chunks_dir):
        os.makedirs(d, exist_ok=True)

    broll_assets, scene_assets = libraries if libraries else load_libraries()
    broll_lookup = {a["id"]: a for a in broll_assets}
    scene_lookup = {a["id"]: a for a in scene_assets}

    state = load_run_state(run_dir)
    if state.get("chunks"):
        print(f"📂 Resuming '{video_name}' ({run_dir})")
    else:
        print(f"🆕 Starting fresh v3 run for '{video_name}' at {run_dir}")

    chunks_json_path = os.path.join(run_dir, "chunks.json")
    if state.get("chunks_json") and os.path.isfile(state["chunks_json"]):
        chunks_json = state["chunks_json"]
    else:
        chunks_json = split_text_for_narration(text, minutes=minutes_per_chunk, output_file=chunks_json_path)
        state["chunks_json"] = chunks_json
        save_run_state(run_dir, state)

    chunks = load_chunks(chunks_json)
    chunk_video_paths = []
    chunks_done_this_call = 0

    for chunk in chunks:
        chunk_id = str(chunk["chunk"])
        chunk_text = chunk["text"]
        c_state = state["chunks"].setdefault(chunk_id, {})

        chunk_video_path = os.path.join(chunks_dir, f"{video_name}-chunk{chunk_id}.mp4")
        if c_state.get("chunk_video_done") and os.path.isfile(chunk_video_path):
            chunk_video_paths.append(chunk_video_path)
            continue

        if chunks_done_this_call >= max_new_chunks:
            # budget for this call is used up — leave the rest for the next turn
            print(f"   ⏸ chunk budget ({max_new_chunks}) reached for '{video_name}' — "
                  f"switching to next file this turn")
            return {
                "status": "partial",
                "chunks_done_this_call": chunks_done_this_call,
                "total_chunks": len(chunks),
                "done_chunks": sum(1 for c in state["chunks"].values() if c.get("chunk_video_done")),
            }

        print(f"\n=== {video_name} — chunk {chunk_id} ===")

        if c_state.get("script"):
            script = c_state["script"]
        else:
            script = create_novel_video_script(chunk_text, language=language)
            c_state["script"] = script
            save_run_state(run_dir, state)

        narration_path = os.path.join(audio_dir, f"{video_name}-chunk{chunk_id}.wav")
        if not (c_state.get("narration_done") and os.path.isfile(narration_path)):
            text_to_audio_gemini(script, narration_path)
            c_state["narration_done"] = True
            save_run_state(run_dir, state)
        duration = get_audio_duration(narration_path)

        if c_state.get("plan"):
            segments = c_state["plan"]["segments"]
            print(f"   ✔ edit-list plan already exists ({len(segments)} segments) — reusing")
        else:
            raw_plan = plan_video(script, duration, broll_assets, scene_assets, broll_ratio, language)
            plan = reconcile_plan(raw_plan, duration, broll_lookup, scene_lookup, broll_ratio)
            segments = plan["segments"]
            c_state["plan"] = plan
            save_run_state(run_dir, state)

        c_state.setdefault("resolved", {})
        for seg in segments:
            order = str(seg["order"])
            if order in c_state["resolved"]:
                seg["path"] = c_state["resolved"][order]["path"]
                seg["source_type"] = c_state["resolved"][order]["source_type"]
                continue

            if seg["source_type"] in ("broll", "stock"):
                lookup = broll_lookup if seg["source_type"] == "broll" else scene_lookup
                asset = lookup.get(seg["asset_id"])
                resolved = {"source_type": seg["source_type"], "path": asset["path"]}
            else:
                img_out = os.path.join(images_dir, f"{video_name}-c{chunk_id}-seg{order}")
                print(f"   segment {order} [generate]: {seg['prompt'][:90]}...")
                result = generate_verified_image_v3(seg["prompt"], img_out, broll_lookup, scene_lookup)
                resolved = {"source_type": result["source_type"], "path": result["path"]}

            seg["path"] = resolved["path"]
            seg["source_type"] = resolved["source_type"]
            c_state["resolved"][order] = resolved
            save_run_state(run_dir, state)

        chunk_video_path = assemble_chunk_video_v3(segments, narration_path, chunk_video_path)
        print(f"   ⏳ confirming chunk {chunk_id} reached Google Drive before continuing...")
        if wait_for_upload(chunk_video_path, label=f"{video_name} chunk {chunk_id}"):
            print(f"   ☁️  chunk {chunk_id} confirmed on Drive")
        c_state["chunk_video_done"] = True
        save_run_state(run_dir, state)
        chunk_video_paths.append(chunk_video_path)
        chunks_done_this_call += 1

    # every chunk is done -> final assembly
    final_path = os.path.join(run_dir, f"{video_name}_final.mp4")
    if state.get("final_done") and os.path.isfile(final_path):
        return {"status": "complete", "final_path": final_path, "chunks_done_this_call": chunks_done_this_call}

    # need every chunk video path, not just the ones made this call
    all_chunk_video_paths = [
        os.path.join(chunks_dir, f"{video_name}-chunk{c['chunk']}.mp4") for c in chunks
    ]
    final_path = assemble_final_video(all_chunk_video_paths, final_path)
    print(f"   ⏳ confirming final video reached Google Drive...")
    if wait_for_upload(final_path, label=f"{video_name} final"):
        print(f"   ☁️  final video confirmed on Drive")
    state["final_done"] = True
    save_run_state(run_dir, state)
    print(f"\n🎬 Done: {final_path}")
    return {"status": "complete", "final_path": final_path, "chunks_done_this_call": chunks_done_this_call}
