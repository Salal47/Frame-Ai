"""
Entry point for the hourly GitHub Actions run.

Order of operations, matching your instructions:
  1. Sync the master registry CSV against SOURCE_DIR — new files get added
     as 'pending'; files already marked 'complete' are left alone and never
     reprocessed.
  2. Round-robin across every incomplete file, processing
     CHUNKS_PER_FILE_PER_TURN chunks at a time per file before switching to
     the next one (only actually rotates when 2+ files are incomplete).
  3. If Gemini exhausts every fallback model for a task, stop the whole run
     immediately (progress already saved) and wait for the next hourly
     schedule — don't crash, don't spin.

(Instagram upload step is disabled for now — see pipeline/instagram_uploader.py.bak
if/when it should come back.)
"""
import sys
import time

from pipeline import config, registry
from pipeline.gemini_manager import AllModelsExhaustedError
from pipeline.library import load_libraries
from pipeline.orchestrator import process_file_step


def run_generation():
    rows = registry.sync_registry()
    queue = registry.incomplete_files_by_rotation(rows)

    if not queue:
        print("✅ Every source file in SOURCE_DIR is already complete. Nothing to generate.")
        return

    print(f"📋 {len(queue)} file(s) still incomplete: {queue}")

    # Loaded once and reused across every file/chunk this run — the libraries
    # rarely change mid-run and reloading per chunk would just re-read every
    # des/*.csv and re-stat every clip for nothing.
    libraries = load_libraries()

    deadline = time.monotonic() + config.MAX_RUN_MINUTES * 60
    i = 0
    any_incomplete_left = True

    while any_incomplete_left and time.monotonic() < deadline:
        any_incomplete_left = False
        # re-fetch the rotation order each lap so a file that just completed
        # drops out and freshly-added files join in
        rows = registry.load_registry()
        queue = registry.incomplete_files_by_rotation(rows)
        if not queue:
            break

        for file_path in queue:
            if time.monotonic() >= deadline:
                print("⏰ MAX_RUN_MINUTES reached — stopping for this run.")
                return
            try:
                result = process_file_step(
                    file_path,
                    max_new_chunks=config.CHUNKS_PER_FILE_PER_TURN,
                    language="Urdu",
                    libraries=libraries,
                )
            except AllModelsExhaustedError as err:
                print(f"\n🛑 {err}")
                print("   Every important Gemini model is rate-limited/exhausted. "
                      "Stopping this run — progress is saved, next hourly schedule will resume.")
                return
            except Exception as err:
                print(f"❌ error processing {file_path}: {err}")
                registry.mark_error(rows, file_path, err)
                continue

            if result["status"] == "complete":
                registry.mark_complete(rows, file_path, result.get("final_path", ""))
                print(f"🎉 {file_path} fully complete -> {result.get('final_path')}")
            else:
                any_incomplete_left = True
                registry.mark_progress(
                    rows, file_path,
                    done_chunks=result["done_chunks"], total_chunks=result["total_chunks"],
                )
                # only actually "switch" if there's another file waiting —
                # with a single incomplete file this loop just continues on it
                if len(queue) > 1:
                    print(f"➡️  switching source file (rotation, {len(queue)} in progress)")

        i += 1


if __name__ == "__main__":
    print("=" * 60)
    print("Insta Uploader v3.3 — scheduled run")
    print("=" * 60)
    run_generation()
    print("✅ Run finished.")
    sys.exit(0)
