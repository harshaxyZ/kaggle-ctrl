"""
test_failover.py - prove the router fails over correctly without a GPU.

Uses a local HTTP server that mimics llama-server's OpenAI API, so the router
can be exercised against real HTTP with real failures. Covers:

  1. a healthy endpoint answers
  2. one endpoint dies -> traffic moves to the other with no client change
  3. the dead endpoint is benched and skipped for later requests
  4. all endpoints dead -> 503 with a clear error, and recovery is requested
  5. health survives a router restart (persisted to disk)
  6. per-key usage is recorded against the key that made the call

Run:  python test_failover.py
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import requests
MODE = {'a': 'ok', 'b': 'ok'}
HITS = {'a': 0, 'b': 0}

class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        name = self.server.name
        HITS[name] += 1
        if MODE[name] == 'dead':
            self._json(502, {'error': 'tunnel gone'})
            return
        if self.path.endswith('/v1/models'):
            self._json(200, {'object': 'list', 'data': [{'id': name}]})
            return
        self._json(404, {'error': 'nope'})

    def do_POST(self):
        name = self.server.name
        HITS[name] += 1
        if MODE[name] == 'dead':
            self._json(502, {'error': 'tunnel gone'})
            return
        if MODE[name] == 'wedged':
            time.sleep(30)
            return
        n = int(self.headers.get('Content-Length', 0))
        _ = self.rfile.read(n)
        self._json(200, {'id': 'cmpl-' + name, 'object': 'chat.completion', 'model': name, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'PONG from ' + name}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})

def start_server(name):
    srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    srv.name = name
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return (srv, 'http://127.0.0.1:%d' % srv.server_address[1])

def fake_deployment(dep_id, account, endpoint, api_key):
    """A deployment dict shaped like server_manager's output."""
    return {'id': dep_id, 'account_id': 'acct-' + account, 'account_name': account, 'model_key': 'configured-model', 'model_spec': {'key': 'configured-model', 'label': 'test'}, 'kernel_ref': 'u/serve-configured-model-' + dep_id[-8:], 'status': 'RUNNING', 'endpoint': endpoint, 'api_key': api_key, 'model_id': 'configured-model.gguf', 'accelerator': 'NvidiaTeslaT4', 'started_at': time.time(), 'uptime_s': 100, 'last_update': time.time(), 'message': '', 'logs_tail': '', 'phase': ''}

def main():
    import router
    import server_manager as sm
    ok = True

    def check(name, cond, extra=''):
        nonlocal ok
        print(('  PASS  ' if cond else '  FAIL  ') + name + ('  ' + extra if extra else ''))
        if not cond:
            ok = False
    srv_a, url_a = start_server('a')
    srv_b, url_b = start_server('b')
    print('fake endpoints:', url_a, url_b)
    with router._LOCK:
        router._health.clear()
        router._stats.clear()
        router._rr_index.clear()
    sm.DEPLOYMENTS.clear()
    sm.DEPLOYMENTS['dep-aaaaaaaa'] = sm.Deployment(id='dep-aaaaaaaa', account_id='acct-a', account_name='a', model_key='configured-model', kernel_ref='u/serve-configured-model-aaaaaaaa', status='RUNNING', endpoint=url_a, api_key='key-a', model_id='configured-model.gguf')
    sm.DEPLOYMENTS['dep-bbbbbbbb'] = sm.Deployment(id='dep-bbbbbbbb', account_id='acct-b', account_name='b', model_key='configured-model', kernel_ref='u/serve-configured-model-bbbbbbbb', status='RUNNING', endpoint=url_b, api_key='key-b', model_id='configured-model.gguf')
    payload = {'model': 'configured-model', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8}
    print('\n[1] both endpoints healthy -> request succeeds')
    st, body, served = router.chat(dict(payload))
    check('status 200', st == 200, str(st))
    check('routed account reported', served.get('account_name') in ('a', 'b'), str(served.get('account_name')))
    check('content returned', 'PONG' in json.dumps(body))
    print('\n[2] endpoint A dies -> traffic moves to B with no client change')
    MODE['a'] = 'dead'
    router._health.clear()
    for _ in range(3):
        st, body, served = router.chat(dict(payload))
    check('request still succeeds', st == 200, str(st))
    check('served by the live account', served.get('account_name') == 'b', str(served.get('account_name')))
    print('\n[3] dead endpoint is benched and skipped')
    check('A benched', not router._is_healthy('dep-aaaaaaaa'))
    check('B still healthy', router._is_healthy('dep-bbbbbbbb'))
    print('\n[4] all endpoints dead -> 503 with a clear error + recovery requested')
    import keepalive
    MODE['b'] = 'dead'
    keepalive.clear_recovery_request()
    for _ in range(3):
        st, body, served = router.chat(dict(payload))
    check('status 503', st == 503, str(st))
    check('error type reported', (body.get('error') or {}).get('type') in ('all_endpoints_failed', 'no_healthy_endpoint'), str((body.get('error') or {}).get('type')))
    check('recovery requested', keepalive.recovery_requested())
    print('\n[5] a dead endpoint is never chosen while a live one exists')
    MODE['b'] = 'ok'
    router._health.clear()
    MODE['a'] = 'dead'
    chosen = set()
    for _ in range(6):
        st, body, served = router.chat(dict(payload))
        if served:
            chosen.add(served.get('account_name'))
    check('only the healthy account served', chosen == {'b'}, str(chosen))
    print('\n[6] health is persisted so a router restart keeps the breaker')
    router._save_health()
    check('health file written', os.path.exists(router.HEALTH_FILE))
    router._health.clear()
    router._load_health()
    check('A still benched after reload', not router._is_healthy('dep-aaaaaaaa'))
    print('\n[7] per-key usage is recorded')
    try:
        import keys
        keys._save([])
        router.chat(dict(payload), api_key='test-key-123')
        rows = keys._load()
        hit = [r for r in rows if r.get('key') == 'test-key-123']
        check('usage attributed to the key', bool(hit) and hit[0].get('requests', 0) >= 1, str(hit[:1]))
    except Exception as e:
        check('usage attributed to the key', False, repr(e)[:80])
    print('\n[8] streaming: a dead endpoint is skipped, the live one streams')
    MODE['a'] = 'dead'
    MODE['b'] = 'ok'
    router._health.clear()
    out = b''.join(router.chat_stream(dict(payload)))
    check('stream produced data', b'data:' in out, str(out[:120]))
    srv_a.shutdown()
    srv_b.shutdown()
    print('\n' + ('ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'))
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())
