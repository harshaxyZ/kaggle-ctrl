"""
cli.py - one command surface for the whole tool.

    python cli.py                  status
    python cli.py up [model]       start a kernel on the best account
    python cli.py down             stop everything
    python cli.py accounts         account pool
    python cli.py add NAME TOKEN [USER]
    python cli.py rm ACCOUNT_ID
    python cli.py endpoint         router url + key
    python cli.py models           self-host catalogue
    python cli.py proxy            live proxy models
    python cli.py quota            GPU and proxy spend
    python cli.py key              mint a dashboard key
    python cli.py serve            run the dashboard
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import json
import os
import sys
import accounts
import keys
import kaggle_client as kc

def _boot_token() -> None:
    """Make sure a Kaggle token is available.

    Order: KAGGLE_API_TOKEN env var, then ~/.kaggle/access_token. There is no
    hardcoded fallback on purpose - a token baked into the repo is a secret in
    git history, and this repo is meant to be a reusable template.
    """
    if not os.environ.get('KAGGLE_API_TOKEN'):
        p = os.path.expanduser('~/.kaggle/access_token')
        if os.path.exists(p):
            os.environ['KAGGLE_API_TOKEN'] = open(p, encoding='utf-8').read().strip()
        else:
            raise SystemExit('No Kaggle token. Set KAGGLE_API_TOKEN or write it to ~/.kaggle/access_token (see .env.example).')

def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))

def main() -> None:
    _boot_token()
    args = sys.argv[1:]
    cmd = args[0] if args else 'status'
    if cmd == 'serve':
        import uvicorn
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', '8787'))
        print(f'dashboard -> http://{host}:{port}')
        print(f'router    -> http://{host}:{port}/v1')
        uvicorn.run('app:app', host=host, port=port, log_level='info')
        return
    import server_manager as sm
    if cmd == 'status':
        _print(sm.state())
    elif cmd in ('up', 'start'):
        _print(sm.start(model_key=args[1] if len(args) > 1 else None))
    elif cmd in ('down', 'stop'):
        _print(sm.stop(None))
    elif cmd == 'endpoint':
        live = sm.state()['live_endpoints']
        _print({'router_base_url': f"http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8787')}/v1", 'note': 'use a dashboard key as the api key', 'tunnels': [{'account': d['account_name'], 'model': d['model_id'], 'url': d['endpoint'] + '/v1'} for d in live]})
    elif cmd == 'models':
        _print([m.as_dict() for m in sm.MODELS.values()])
    elif cmd == 'accounts':
        _print({'accounts': [a.as_dict() for a in accounts.load()], 'pool': accounts.pool_summary()})
    elif cmd == 'add' and len(args) >= 3:
        a = accounts.add(args[1], args[2], args[3] if len(args) > 3 else '')
        _print({'ok': True, 'account': a.as_dict()})
    elif cmd == 'rm' and len(args) >= 2:
        _print({'ok': accounts.remove(args[1])})
    elif cmd == 'quota':
        _print({'gpu': kc.get_accelerator_quota().as_dict(), 'proxy': kc.get_proxy_quota().as_dict()})
    elif cmd == 'proxy':
        live = [m for m in kc.get_proxy_catalogue() if m.available]
        _print([m.as_dict() for m in live])
    elif cmd == 'key':
        _print(keys.generate(args[1] if len(args) > 1 else 'cli'))
    elif cmd in ('help', '-h', '--help'):
        print(__doc__)
    else:
        print(__doc__)
if __name__ == '__main__':
    main()
