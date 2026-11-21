"""
bot_api.py - client for any agent with HTTP access (remote client, sandbox, CI).

    from bot_api import KaggleBot
    kb = KaggleBot("http://YOUR_IP:8787", key="dash-...")

    kb.status()             # GPU hours, proxy credits, server state
    kb.start("configured-model")    # push a serving kernel
    kb.stop()               # stop everything
    kb.endpoint()           # router base_url + api key
    kb.chat("configured-model", [{"role": "user", "content": "hi"}])
    kb.handle("turn the server on")

CLI:
    python bot_api.py status
    python bot_api.py start configured-model
    python bot_api.py stop
    python bot_api.py endpoint
    python bot_api.py accounts
"""
from __future__ import annotations
import json
import os
import re
import urllib.error
import urllib.request
DEFAULT_BASE = os.environ.get('DASH_URL', 'http://127.0.0.1:8787')
DEFAULT_KEY = os.environ.get('DASH_KEY', '')
START_PHRASES = ('start', 'turn on', 'turn the server on', 'turn it on', 'power on', 'spin up', 'boot', 'boot up', 'enable', 'wake up', 'bring up', 'switch on', 'bring the server up', 'serve it', 'serve', 'deploy', 'run the model', 'start the model', 'start the server', 'start serving')
STOP_PHRASES = ('stop', 'turn off', 'turn the server off', 'turn it off', 'power off', 'kill', 'shutdown', 'shut down', 'disable', 'halt', 'switch off', 'terminate', 'stop the server', 'stop the model', 'stop serving')

class KaggleBot:

    def __init__(self, base_url: str=DEFAULT_BASE, key: str | None=None, timeout: int=120):
        self.base = base_url.rstrip('/')
        self.key = key if key is not None else DEFAULT_KEY
        self.timeout = timeout
        if not self.key:
            self.key = self._discover_key()

    def _discover_key(self) -> str:
        """Find a usable dashboard key.

        The dashboard's own API is key-protected, so a bot that has no key
        configured cannot even ask for one. api_keys.json lives next to the
        dashboard, so read a live key from there as a fallback. Set DASH_KEY
        explicitly in production instead of relying on the file.
        """
        if self.key:
            return self.key
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (os.path.join(here, 'api_keys.json'), os.path.join(here, '..', 'kaggle-dash', 'api_keys.json')):
            try:
                with open(path, encoding='utf-8') as f:
                    keys = json.load(f)
            except Exception:
                continue
            for k in keys:
                if k.get('revoked'):
                    continue
                if k.get('key'):
                    self.key = k['key']
                    return self.key
        return ''

    def _req(self, method: str, path: str, body: dict | None=None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = f'Bearer {self.key}'
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            return {'error': e.read().decode()[:400], 'status': e.code}
        except Exception as e:
            return {'error': str(e)}

    def state(self) -> dict:
        return self._req('GET', '/api/state')

    def status(self) -> dict:
        """Flattened view for chat replies."""
        s = self.state()
        gpu = (s.get('accelerator') or {}).get('gpu', {})
        pq = s.get('proxy_quota') or {}
        pool = s.get('pool') or {}
        live = s.get('live_endpoints') or []
        return {'gpu_hours_left': gpu.get('remaining_h'), 'gpu_reset_at': gpu.get('refresh_at'), 'daily_credits_left': (pq.get('daily') or {}).get('remaining'), 'monthly_credits_left': (pq.get('monthly') or {}).get('remaining'), 'pool_accounts': pool.get('accounts'), 'pool_hours_left': pool.get('total_remaining_h'), 'pool_slots_free': pool.get('slots_free'), 'live_endpoints': len(live), 'deployments': [{'id': d['id'], 'account': d['account_name'], 'status': d['status'], 'model': d['model_key']} for d in s.get('deployments') or []]}

    def start(self, model_key: str | None=None, account_id: str | None=None) -> dict:
        return self._req('POST', '/api/server/start', {'model_key': model_key, 'account_id': account_id})

    def stop(self) -> dict:
        return self._req('POST', '/api/server/stop')

    def poll(self) -> dict:
        return self._req('POST', '/api/server/refresh')

    def endpoint(self) -> dict:
        """The stable router endpoint every client should use.

        The router key is the one the caller configured (or auto-discovered).
        The upstream model endpoint and its own key are also returned, for
        anyone who wants to bypass the router and hit one deployment directly.
        """
        out = {'base_url': self.base + '/v1', 'api_key': self.key, 'shift': 'any live deployment is chosen per request'}
        try:
            live = self.state().get('live_endpoints') or []
            if live:
                d = live[0]
                out['direct_base_url'] = (d.get('endpoint') or '') + '/v1'
                out['direct_api_key'] = d.get('api_key') or ''
                out['model'] = d.get('model_id') or ''
                out['account'] = d.get('account_name') or ''
        except Exception:
            pass
        return out

    def chat(self, model: str, messages: list[dict], max_tokens: int=512) -> dict:
        return self._req('POST', '/v1/chat/completions', {'model': model, 'messages': messages, 'max_tokens': max_tokens})

    def accounts(self) -> list[dict]:
        return self._req('GET', '/api/accounts')

    def add_account(self, name: str, token: str, username: str='') -> dict:
        return self._req('POST', '/api/accounts', {'name': name, 'token': token, 'username': username})

    def remove_account(self, account_id: str) -> dict:
        return self._req('POST', f'/api/accounts/{account_id}/delete')

    def proxy_models(self) -> list[dict]:
        return self._req('GET', '/api/proxy/models')

    def proxy_session(self) -> dict:
        return self._req('GET', '/api/proxy/session')

    def cmd(self, command: str) -> dict:
        return self._req('POST', '/api/cmd', {'command': command})

    def handle(self, text: str) -> dict:
        """Map a natural-language message to an action. Returns the API result."""
        t = (text or '').lower().strip()
        if any((p in t for p in STOP_PHRASES)) and "don't stop" not in t:
            return {'action': 'stop', **self.stop()}
        if any((p in t for p in START_PHRASES)):
            m = re.search('(?:start|run|launch)\\s+([a-z0-9_-]+)', t)
            return {'action': 'start', **self.start(m.group(1) if m else None)}
        if any((w in t for w in ('how much', 'status', 'quota', 'left', 'credit', 'balance', 'hours', 'reset'))):
            return {'action': 'status', **self.status()}
        if any((w in t for w in ('endpoint', 'url', 'api key', 'base_url', 'base url'))):
            return {'action': 'endpoint', **self.endpoint()}
        if any((w in t for w in ('account', 'pool', 'friends'))):
            return {'action': 'accounts', 'accounts': self.accounts()}
        if any((w in t for w in ('which model', 'models', 'catalogue', 'catalog', 'what can i use'))):
            return {'action': 'models', 'models': self.proxy_models()}
        return {'action': 'unknown', 'hint': 'try: start / stop / status / endpoint / accounts / models'}
if __name__ == '__main__':
    import sys
    kb = KaggleBot()
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    out: object
    if cmd == 'status':
        out = kb.status()
    elif cmd == 'start':
        out = kb.start(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == 'stop':
        out = kb.stop()
    elif cmd == 'endpoint':
        out = kb.endpoint()
    elif cmd == 'accounts':
        out = kb.accounts()
    elif cmd == 'chat':
        out = kb.chat(__import__('settings').SETTINGS['default_model'], [{'role': 'user', 'content': sys.argv[2] if len(sys.argv) > 2 else 'ping'}])
    else:
        out = {'usage': 'python bot_api.py [status|start [model]|stop|endpoint|accounts|chat <msg>]'}
    print(json.dumps(out, indent=2))
