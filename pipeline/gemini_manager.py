"""
Gemini API key + model fallback manager. Logic is unchanged from the
v3.3 notebook — only the exhaustion signal is upgraded from a generic
RuntimeError to AllModelsExhaustedError so main.py can catch it precisely
and stop the whole run cleanly (requirement: if every important Gemini
model hits its limit, stop and wait for the next hourly schedule).
"""
import random
import time
import json
import os
from google import genai

from . import config

MODEL_STATE_PATH = os.path.join(config.DRIVE_ROOT, "model_state.json")


class AllModelsExhaustedError(RuntimeError):
    """Raised when every model in a task's fallback chain is 404/rate-limited."""
    def __init__(self, task, chain):
        self.task = task
        self.chain = chain
        super().__init__(
            f"All fallback models exhausted for task '{task}': {chain}. "
            "Every model either 404'd or stayed rate-limited/quota-exhausted."
        )


def _load_model_state():
    if os.path.isfile(MODEL_STATE_PATH):
        try:
            with open(MODEL_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_model_state(state):
    os.makedirs(os.path.dirname(MODEL_STATE_PATH), exist_ok=True)
    with open(MODEL_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


class GeminiKeyManager:
    """
    Holds a pool of Gemini API keys AND, per task, a fallback chain of models.

    - Keys are picked at random per call; a rate-limited key is retried on a
      different key, an invalid/denied key is dropped for the session.
    - Models are looked up per task from MODEL_FALLBACKS. If the current
      model for a task 404s or stays rate-limited across all keys for two
      full rounds, the manager advances to the next model in that task's
      chain and persists the position to model_state.json on Drive.
    - If every model in the chain is exhausted, raises AllModelsExhaustedError.
    """

    def __init__(self, api_keys, model_fallbacks):
        real_keys = [k for k in api_keys if k]
        if not real_keys:
            raise ValueError("No Gemini API keys provided.")
        self.api_keys = real_keys
        self.disabled_keys = set()
        self._clients = {k: genai.Client(api_key=k) for k in real_keys}
        self.model_fallbacks = model_fallbacks
        self.state = _load_model_state()

    def current_model(self, task):
        chain = self.model_fallbacks[task]
        idx = self.state.get(task, {}).get("index", 0)
        idx = min(idx, len(chain) - 1)
        return chain[idx], idx

    def _advance_model(self, task):
        chain = self.model_fallbacks[task]
        _, idx = self.current_model(task)
        if idx + 1 >= len(chain):
            raise AllModelsExhaustedError(task, chain)
        new_idx = idx + 1
        self.state[task] = {"index": new_idx, "model": chain[new_idx]}
        _save_model_state(self.state)
        print(f"   🔁 switching task '{task}' -> model '{chain[new_idx]}' "
              f"(saved to model_state.json — future runs start here)")

    def _random_key(self, exclude):
        live = [k for k in self.api_keys if k not in self.disabled_keys]
        pool = [k for k in live if k not in exclude] or live or self.api_keys
        return random.choice(pool)

    @staticmethod
    def _classify(err):
        msg = str(err)
        if "404" in msg or "NOT_FOUND" in msg:
            return "not_found"
        if any(s in msg for s in ("403", "401", "PERMISSION_DENIED", "UNAUTHENTICATED", "denied access")):
            return "key_invalid"
        if any(s in msg.lower() for s in ("429", "resource_exhausted", "rate limit", "quota", "unavailable", "503")):
            return "rate_limit"
        return "other"

    def call(self, task, fn):
        while True:
            model, _ = self.current_model(task)
            tried = set()
            sleep_secs = config.RATE_LIMIT_SLEEP_SECS
            rounds_exhausted = 0
            attempt = 0

            while attempt < config.MAX_RETRIES_PER_KEY_ROUND:
                attempt += 1
                key = self._random_key(tried)
                tried.add(key)
                try:
                    return fn(self._clients[key], model)
                except Exception as err:
                    kind = self._classify(err)

                    if kind == "not_found":
                        print(f"   ✗ model '{model}' not available for your project (404).")
                        break

                    if kind == "key_invalid":
                        print(f"❌ key ...{key[-4:]} denied/invalid — disabling for this session")
                        self.disabled_keys.add(key)
                        if len(self.disabled_keys) >= len(self.api_keys):
                            raise RuntimeError(f"All API keys are invalid/denied: {err}") from err
                        attempt -= 1
                        continue

                    if kind == "rate_limit":
                        live_count = len(self.api_keys) - len(self.disabled_keys)
                        print(f"⚠️ rate limit on '{model}' / key ...{key[-4:]} "
                              f"(attempt {attempt}/{config.MAX_RETRIES_PER_KEY_ROUND})")
                        if len({k for k in tried if k not in self.disabled_keys}) >= max(live_count, 1):
                            rounds_exhausted += 1
                            tried = set()
                            if rounds_exhausted >= 2:
                                break
                            print(f"   all live keys rate-limited on '{model}' — sleeping {sleep_secs:.0f}s...")
                            time.sleep(sleep_secs)
                            sleep_secs *= config.RATE_LIMIT_BACKOFF
                        continue

                    print(f"⚠️ Gemini call failed ({type(err).__name__}: {err}) — retrying...")
                    time.sleep(2)
            else:
                # attempt budget exhausted without an explicit break -> try next model too
                pass

            self._advance_model(task)


key_manager = GeminiKeyManager(config.GEMINI_API_KEYS, config.MODEL_FALLBACKS)


def gemini_text(task, contents, config=None):
    # NOTE: parameter is named `config` (shadowing the config module inside
    # this function only) to match every call site copied verbatim from the
    # notebook, e.g. gemini_text("light", prompt, config=types.GenerateContentConfig(...))
    generate_content_config = config
    return key_manager.call(task, lambda client, model: client.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    ))
