"""
Confirms a file has actually finished uploading to Google Drive before the
pipeline moves on, instead of trusting that "file exists locally" == "file
is safe on Drive".

Why this is needed: the workflow mounts Drive with `--vfs-cache-mode writes`,
which caches writes on the runner's local disk and uploads them to Drive
*asynchronously* in the background (a few seconds after the file is closed,
per --vfs-write-back). That's fine when the run finishes normally, but if the
job gets killed or times out right after a chunk is produced, that chunk can
look "done" to the pipeline (the local cache file exists) while the bytes
never actually reached Drive, silently losing progress.

wait_for_upload() polls rclone's `--rc` API (enabled in the workflow's mount
step) and blocks until the given path is no longer in rclone's pending
write-back queue, i.e. it has actually been uploaded.
"""
import os
import time

import requests

RC_ADDR = os.environ.get("RCLONE_RC_ADDR", "http://127.0.0.1:5572")
POLL_SECS = 2
DEFAULT_TIMEOUT = 300  # generous — chunk videos can be tens of MB on a slow link


def wait_for_upload(path, timeout=DEFAULT_TIMEOUT, label=None):
    """
    Block until `path` is no longer pending upload in rclone's VFS
    write-back queue (i.e. it has reached Google Drive).

    Returns True if confirmed uploaded, False if we couldn't confirm
    (rc endpoint unreachable — e.g. running locally without the mount — or
    timed out). False is a "couldn't verify" signal, not necessarily a
    failure, so callers just log and continue rather than raising.
    """
    tag = label or os.path.basename(path)
    needle = os.path.basename(path)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            resp = requests.post(f"{RC_ADDR}/vfs/queue", timeout=5)
            resp.raise_for_status()
            queue = resp.json().get("queue") or []
        except Exception:
            # rc API not reachable — most likely running outside the GH
            # Actions mount (e.g. local testing). Nothing to confirm against.
            return False

        pending = [q for q in queue if needle in q.get("name", "")]
        if not pending:
            return True
        time.sleep(POLL_SECS)

    print(f"   ⚠️  couldn't confirm '{tag}' finished uploading to Drive "
          f"within {timeout}s — continuing, but check the next run's log.")
    return False
