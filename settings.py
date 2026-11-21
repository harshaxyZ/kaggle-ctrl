"""Per-install settings. Runtime secrets and model choices stay out of version control."""
import json
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_settings():
    path = Path(os.environ.get('KGCTRL_CONFIG', str(ROOT / '.local/settings.json')))
    data = json.loads(path.read_text()) if path.exists() else {}
    models = data.get('models')
    if not models:
        raise ValueError('No models configured. Run python setup_local.py first.')
    if not isinstance(models, list) or not all(isinstance(m, dict) and m.get('key') for m in models):
        raise ValueError('models must be a list of model objects with unique keys')
    if len({m['key'] for m in models}) != len(models):
        raise ValueError('Model keys must be unique')
    default = data.get('default_model', models[0]['key'])
    if default not in {m['key'] for m in models}:
        raise ValueError('default_model must name a configured model')
    return {**data, 'models': models, 'default_model': default}

SETTINGS = load_settings()

def installation_id():
    path = ROOT / '.local/installation-id'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(16))
        path.chmod(0o600)
    return path.read_text().strip()
