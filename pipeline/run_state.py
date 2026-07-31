"""Per-source-file resumable checkpoint (run_state.json inside each run_dir).
Unchanged logic from cell 26 of the notebook."""
import json
import os


def _run_state_path(run_dir):
    return os.path.join(run_dir, "run_state.json")


def load_run_state(run_dir):
    """Load this source file's checkpoint from Drive, if one exists."""
    path = _run_state_path(run_dir)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks_json": None, "chunks": {}, "final_done": False}


def save_run_state(run_dir, state):
    os.makedirs(run_dir, exist_ok=True)
    with open(_run_state_path(run_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
