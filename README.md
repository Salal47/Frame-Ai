# Insta Uploader v3.3 — GitHub Actions deployment

This converts the `insta_uploader_v3_3.ipynb` Colab notebook into a scheduled
GitHub Actions pipeline. Every hour it:

1. Scans your Drive `sources/` folder for new text/PDF/DOCX files.
2. Round-robins between every unfinished file, doing 2–3 chunks per file per
   turn (only switches files if 2+ are actually incomplete).
3. Stops immediately — without crashing — if Gemini's quota is exhausted
   across every fallback model, and waits for the next hourly run.
4. ~~Uploads any freshly-finished video to Instagram~~ — **disabled for now**
   (see below).
5. Never runs two copies at once — a second scheduled trigger queues behind
   whichever run is already in progress instead of running in parallel.

## What maps to what (your requirements)

| Requirement | Where |
|---|---|
| CSV that tracks which files are done so completed ones aren't reprocessed | `pipeline/registry.py` → `source_registry.csv` on Drive |
| Switch file every 2–3 chunks when 2+ files available | `main.py` `run_generation()` + `CHUNKS_PER_FILE_PER_TURN` |
| Stop entirely when all important Gemini models hit their limit, wait for next schedule | `pipeline/gemini_manager.py` (`AllModelsExhaustedError`) caught in `main.py` |
| Run every 1 hour | `.github/workflows/pipeline.yml` → `cron: "0 * * * *"` |
| Instagram section: upload / get / set / output / generations | `pipeline/instagram_uploader.py.bak` — **disabled for now**, see "Re-enabling Instagram upload" below |
| B-roll clips from `My Drive/Content Gen/Media/b-roll` | `pipeline/config.py` → `BROLL_DIR` (unchanged path) |
| Handled through API | Gemini via `google-genai`, Instagram via `instagrapi` |
| One action at a time, no parallel runs | `concurrency:` block in the workflow |

## One-time setup

### 1. Google Drive access (service account, no login prompt)

GitHub Actions can't run `drive.mount()` like Colab — it uses `rclone` with a
Google service account instead.

1. In Google Cloud Console, create a service account and download its JSON
   key.
2. In Google Drive, create/locate one parent folder that will hold both
   `InstaUploaderPipeline` and `Content Gen` as subfolders (if they aren't
   already siblings, move or symlink-share them so both sit under one shared
   folder — a service account can only see what's explicitly shared with it,
   it cannot browse your whole "My Drive").
3. Share that parent folder with the service account's email (found inside
   the JSON key, looks like `xxx@xxx.iam.gserviceaccount.com`) as **Editor**.
4. Copy that parent folder's ID from its Drive URL
   (`https://drive.google.com/drive/folders/<THIS_PART>`).
5. Add two GitHub secrets:
   - `GDRIVE_SA_JSON` — paste the full service-account JSON key.
   - `GDRIVE_ROOT_FOLDER_ID` — the folder ID from step 4.

### 2. Gemini API keys

Add secret `GEMINI_API_KEYS` — 2–3 free keys from
<https://aistudio.google.com/apikey>, comma-separated:
`key1,key2,key3`

### 3. (Optional) OpenAI image generation

Only needed if you want `gpt-image-2`/`gptimage` in `IMAGE_MODEL_CHAIN` —
otherwise those two fall back to Pollinations automatically.
- `OPENAI_API_KEY`

### 4. Drop your content in

On Drive, under the shared parent folder:
- `InstaUploaderPipeline/sources/` — put your `.txt`/`.md`/`.pdf`/`.docx`
  source files here. New files are picked up automatically on the next run.
- `Content Gen/Media/b-roll/des/*.csv` and
  `Content Gen/Media/scene_library/des/*.csv` — your existing b-roll/stock
  libraries, unchanged from the notebook's expected layout.

### 5. Enable the workflow

Push this repo to GitHub. The workflow runs automatically on the hourly
schedule; you can also trigger it manually from the Actions tab
(`workflow_dispatch`).

## Re-enabling Instagram upload

The Instagram upload step is fully built but currently disabled. To turn it
back on:

1. Rename `pipeline/instagram_uploader.py.bak` → `pipeline/instagram_uploader.py`.
2. In `main.py`, add back:
   ```python
   from pipeline.instagram_uploader import upload_pending

   def run_instagram_uploads():
       try:
           upload_pending()
       except Exception as err:
           print(f"⚠️ Instagram upload step failed (generation still succeeded): {err}")
   ```
   and call `run_instagram_uploads()` after `run_generation()` in `__main__`.
3. Add `instagrapi` back to `requirements.txt`.
4. Add `IG_USERNAME` / `IG_PASSWORD` GitHub secrets and pass them (plus
   `IG_MAX_UPLOADS_PER_RUN`) as env vars in `.github/workflows/pipeline.yml`'s
   "Run pipeline" step.

## Local test run

```bash
export DRIVE_MOUNT_ROOT=/path/to/a/locally-mounted-or-synced/drive/folder
export GEMINI_API_KEYS=key1,key2
export IG_USERNAME=... IG_PASSWORD=...
pip install -r requirements.txt
python main.py
```

## Notes

- `MAX_RUN_MINUTES` (default 45, set in the workflow) keeps each run under
  the hourly cadence so it always has time to save state before GitHub
  kills the job at `timeout-minutes: 55`.
- `CHUNKS_PER_FILE_PER_TURN` (default 3) controls the file-switch cadence.
- `IG_MAX_UPLOADS_PER_RUN` (default 1) caps how many finished videos get
  posted per hourly run.
- All generation/model-fallback/TTS/image-verification logic is unchanged
  from the v3.3 notebook — only the Colab-only bits (Drive mount, `%pip`,
  `IPython.display`) were replaced with their headless equivalents.
