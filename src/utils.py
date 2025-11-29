import os
import json

def _get_secrets_path() -> str:
    home = os.path.expanduser("~")
    base_dir = os.path.join(home, ".mangatranslator")
    return os.path.join(base_dir, "secrets.json")

def load_local_secrets():
    path = _get_secrets_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_local_secrets(secrets: dict) -> None:
    path = _get_secrets_path()
    base_dir = os.path.dirname(path)
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(secrets, f)
