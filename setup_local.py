"""Interactive first-run setup. Never overwrites an existing local installation."""
from pathlib import Path
import getpass
import json
import os

ROOT = Path(__file__).resolve().parent

def main():
    config = ROOT / '.local/settings.json'
    env = ROOT / '.env'
    if config.exists() or env.exists():
        raise SystemExit('Local configuration already exists. Edit it manually; setup will not overwrite it.')
    username = input('Your Kaggle username: ').strip()
    if not username:
        raise SystemExit('A Kaggle username is required.')
    token = getpass.getpass('Your Kaggle API token (hidden): ').strip()
    if not token or any(c in token for c in '\n\r'):
        raise SystemExit('A valid single-line token is required.')
    repo = input('Your chosen Hugging Face repository (owner/repository): ').strip()
    filename = input('GGUF filename inside that repository: ').strip()
    alias = input('Local model alias (letters, digits and hyphens): ').strip()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', alias) or not re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', repo) or not re.fullmatch(r'[A-Za-z0-9._/-]+', filename):
        raise SystemExit('Check the repository, filename and alias formats.')
    data = json.loads((ROOT / 'settings.example.json').read_text())
    data['default_model'] = alias
    data['models'][0].update(key=alias, label=alias, hf_repo=repo, filename=filename)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(data, indent=2) + '\n')
    config.chmod(0o600)
    env.write_text('KAGGLE_USERNAME=' + username + '\nKAGGLE_API_TOKEN=' + token + '\nHOST=127.0.0.1\nPORT=8787\n')
    env.chmod(0o600)
    print('Saved private local configuration. Review context/hardware settings, then run: python run.py')
    print('Do not upload .env, .local/, account files or generated notebooks.')

if __name__ == '__main__':
    main()
