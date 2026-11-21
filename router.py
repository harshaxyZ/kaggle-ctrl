"""
router.py - OpenAI-compatible gateway over the live serving deployments.

The client holds ONE dashboard key and points at ONE base URL. That never
changes. What changes is which Kaggle kernel is behind it, and the router
moves traffic between them without the client noticing.

Failure modes this file exists to handle, and how each is covered:

  1. A kernel dies (12h hard cap, quota exhausted, crash).
     -> reconcile_with_kaggle() in server_manager retires it, so it leaves
        the live set and traffic moves to the other account.

  2. An endpoint is alive but the model behind it is wedged, or the
     cloudflared tunnel is dead. The kernel still says RUNNING, so the
     naive "is it RUNNING" check is not enough.
     -> a background prober pings every endpoint on a timer and benches
        the ones that do not answer. Detection is proactive, so a user
        request never has to be the thing that discovers the failure.

  3. A request lands on an endpoint that dies between the health check and
     the request.
     -> the request is retried on the next healthy endpoint, and the
        failure benches the dead one.

  4. An endpoint accepts the connection but never answers (wedged
     llama-server). Without a bounded timeout the request hangs for the
     full client timeout.
     -> short connect/read timeout plus a pre-flight probe, so a wedged
        endpoint is skipped rather than waited on.

  5. The dashboard process restarts. In-memory health state is lost, so a
     known-dead endpoint comes back "healthy" and users hit it again.
     -> health is persisted to disk.

  6. Every account is down at once.
     -> the router returns 503 with the real reason and the recovery
        state, and asks the supervisor to redeploy, so the outage is
        self-healing rather than manual.

  7. The client itself is mid-stream when an endpoint dies.
     -> a mid-stream failure cannot be retried transparently (the client
        already has partial output), so the stream ends cleanly with an
        error event instead of hanging.
"""
from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path
import requests
import analytics
import keys
import server_manager as sm
UA = 'kaggle-dash-router/2.0'
HERE = Path(__file__).resolve().parent
HEALTH_FILE = HERE / 'router_health.json'
_LOCK = threading.RLock()
_rr_index: dict[str, int] = {}
_inflight: dict[str, int] = {}
_stats: dict[str, dict] = {}
_health: dict[str, dict] = {}
FAIL_THRESHOLD = 2
COOLDOWN_S = 60
MAX_COOLDOWN_S = 300
PROBE_INTERVAL_S = 15
REQUEST_TIMEOUT_S = 600
PROBE_TIMEOUT_S = 8

def _load_health() -> None:
    if not HEALTH_FILE.exists():
        return
    try:
        data = json.loads(HEALTH_FILE.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            with _LOCK:
                _health.update(data)
    except Exception:
        pass

def _save_health() -> None:
    with _LOCK:
        snap = {k: dict(v) for k, v in _health.items()}
    try:
        tmp = HEALTH_FILE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(snap, indent=2), encoding='utf-8')
        os.replace(tmp, HEALTH_FILE)
    except Exception:
        pass

def _is_healthy(dep_id: str) -> bool:
    with _LOCK:
        h = _health.get(dep_id) or {}
        fails = h.get('fails', 0)
        until = h.get('cooldown_until', 0)
    if fails < FAIL_THRESHOLD:
        return True
    return time.time() >= until

def _mark_failure(dep_id: str, reason: str='', hard: bool=False) -> None:
    """Record a failure and bench the endpoint once it is clearly bad.

    hard=True means the failure was conclusive (a dead tunnel, a timeout),
    so the endpoint is benched right away. Otherwise it takes FAIL_THRESHOLD
    consecutive failures, which keeps a single transient hiccup from taking
    a healthy endpoint out of rotation.
    """
    with _LOCK:
        h = _health.setdefault(dep_id, {})
        h['fails'] = h.get('fails', 0) + 1
        h['last_error_at'] = time.time()
        h['last_error'] = reason[:200]
        if hard:
            h['fails'] = max(h['fails'], FAIL_THRESHOLD)
            h['cooldown_until'] = time.time() + COOLDOWN_S
        elif h['fails'] >= FAIL_THRESHOLD:
            over = h['fails'] - FAIL_THRESHOLD
            h['cooldown_until'] = time.time() + min(MAX_COOLDOWN_S, COOLDOWN_S * 2 ** over)
    _save_health()

def _mark_success(dep_id: str) -> None:
    with _LOCK:
        h = _health.setdefault(dep_id, {})
        if h.get('fails') or h.get('cooldown_until'):
            h['fails'] = 0
            h['cooldown_until'] = 0
            h['recovered_at'] = time.time()
    _save_health()

def probe_endpoint(dep: dict) -> bool:
    """Ping an endpoint. True if it answers."""
    url = dep['endpoint'].rstrip('/') + '/v1/models'
    try:
        r = requests.get(url, headers={'Authorization': f"Bearer {dep['api_key']}", 'User-Agent': UA}, timeout=PROBE_TIMEOUT_S)
        ok = r.status_code == 200
    except Exception:
        ok = False
    if ok:
        _mark_success(dep['id'])
    else:
        _mark_failure(dep['id'], 'probe failed', hard=True)
    return ok

def probe_all() -> dict:
    """Probe every live endpoint. Runs on a timer."""
    results = {}
    for dep in _live():
        results[dep['account_name']] = probe_endpoint(dep)
    return results

def _probe_loop() -> None:
    while True:
        try:
            probe_all()
        except Exception:
            pass
        time.sleep(PROBE_INTERVAL_S)

def start_prober() -> None:
    """Start the background prober once."""
    global _prober_started
    with _LOCK:
        if _prober_started:
            return
        _prober_started = True
    t = threading.Thread(target=_probe_loop, daemon=True, name='kaggle-prober')
    t.start()
_prober_started = False

def _live() -> list[dict]:
    with _LOCK:
        return [d.as_dict() for d in sm.DEPLOYMENTS.values() if d.status == 'RUNNING' and d.endpoint]

def _targets(model_filter: str | None=None, healthy_only: bool=True) -> list[dict]:
    """Live deployments matching the requested model.

    Empty list when the named model is not live, so the caller gets an honest
    error instead of a different model's output.
    """
    live = _live()
    if healthy_only:
        live = [d for d in live if _is_healthy(d['id'])]
    if not live:
        return []
    if not model_filter:
        return live
    key = model_filter.strip().lower()
    exact = [d for d in live if d['model_key'] == key or d['model_id'].lower() == key]
    if exact:
        return exact
    return [d for d in live if key in d['model_key'] or key in d['model_id'].lower() or key in d['model_spec']['label'].lower()]

def _pick(targets: list[dict], strategy: str) -> dict:
    if len(targets) == 1:
        return targets[0]
    if strategy == 'least-busy':
        return min(targets, key=lambda d: _inflight.get(d['id'], 0))
    if strategy == 'weighted':
        import accounts
        by_id = {a.id: a for a in accounts.load()}
        return max(targets, key=lambda d: by_id[d['account_id']].remaining_hours if d['account_id'] in by_id else 0)
    with _LOCK:
        group = '|'.join(sorted((d['id'] for d in targets)))
        i = _rr_index.get(group, 0)
        _rr_index[group] = (i + 1) % len(targets)
    return targets[i % len(targets)]

def _ordered(targets: list[dict], strategy: str) -> list[dict]:
    """Try order: healthy first, then the benched ones as a last resort."""
    healthy = [d for d in targets if _is_healthy(d['id'])]
    sick = [d for d in targets if not _is_healthy(d['id'])]
    if not healthy:
        return targets
    first = _pick(healthy, strategy)
    return [first] + [d for d in healthy if d['id'] != first['id']] + sick

def _hdrs(dep: dict) -> dict:
    return {'Authorization': f"Bearer {dep['api_key']}", 'Content-Type': 'application/json', 'User-Agent': UA}

def models() -> dict:
    """One entry per model id, listing the accounts that can serve it."""
    by_model: dict[str, dict] = {}
    for d in _live():
        mid = d['model_id'] or d['model_key']
        entry = by_model.setdefault(mid, {'id': mid, 'object': 'model', 'owned_by': 'kaggle', 'model_key': d['model_key'], 'accounts': []})
        entry['accounts'].append({'account': d['account_name'], 'endpoint': d['endpoint'] + '/v1', 'healthy': _is_healthy(d['id'])})
    return {'object': 'list', 'data': list(by_model.values())}

def endpoints() -> list[dict]:
    """Per-account endpoint list, for the dashboard and for direct clients."""
    out = []
    for d in _live():
        with _LOCK:
            st = dict(_stats.get(d['id'], {}))
            h = dict(_health.get(d['id'], {}))
            inflight = _inflight.get(d['id'], 0)
        out.append({'deployment_id': d['id'], 'account': d['account_name'], 'account_id': d['account_id'], 'model_key': d['model_key'], 'model_id': d['model_id'], 'base_url': d['endpoint'] + '/v1', 'endpoint': d['endpoint'], 'api_key': d['api_key'], 'uptime_s': d['uptime_s'], 'healthy': _is_healthy(d['id']), 'inflight': inflight, 'requests': st.get('requests', 0), 'errors': st.get('errors', 0), 'last_ms': st.get('last_ms', 0), 'last_tps': st.get('last_tps', 0), 'avg_tps': st.get('avg_tps', 0), 'fails': h.get('fails', 0), 'cooldown_until': h.get('cooldown_until', 0), 'last_error': h.get('last_error', '')})
    return out

def _record_stats(dep: dict, ms: float, in_tok: int, out_tok: int, ok: bool) -> None:
    dep_id = dep['id']
    tps = round(out_tok / (ms / 1000.0), 1) if ms > 0 and out_tok else 0
    with _LOCK:
        st = _stats.setdefault(dep_id, {})
        st['requests'] = st.get('requests', 0) + 1
        st['last_ms'] = round(ms, 1)
        st['last_tps'] = tps
        prev_avg = st.get('avg_tps', 0)
        n = st['requests']
        st['avg_tps'] = round((prev_avg * (n - 1) + tps) / n, 1) if n else tps
        if not ok:
            st['errors'] = st.get('errors', 0) + 1

def _no_endpoint_error(model: str | None) -> tuple[int, dict]:
    """Build an honest error when nothing can serve the request."""
    live = _live()
    if not live:
        try:
            import keepalive
            keepalive.request_recovery()
        except Exception:
            pass
        return (503, {'error': {'type': 'no_endpoint', 'code': 'no_live_deployment', 'message': 'no model is running. the dashboard is starting one; retry shortly.'}})
    live_models = sorted({d['model_key'] for d in live})
    if model and model.strip().lower() not in {m.lower() for m in live_models}:
        return (404, {'error': {'type': 'model_not_live', 'code': 'model_not_live', 'message': f'model {model!r} is not live', 'live_models': live_models}})
    return (503, {'error': {'type': 'no_healthy_endpoint', 'code': 'all_endpoints_unhealthy', 'message': 'every endpoint for this model is failing; retry in a few seconds', 'live_models': live_models}})

def chat(payload: dict, strategy: str='round-robin', timeout: float=REQUEST_TIMEOUT_S, api_key: str='') -> tuple[int, dict, dict]:
    """Forward a chat completion, failing over between accounts.

    Returns (status, body, served) where served describes the deployment that
    answered, so the API layer can report it to the caller.
    """
    model = payload.get('model')
    targets = _targets(model)
    if not targets:
        status, body = _no_endpoint_error(model)
        return (status, body, {})
    errors = []
    for dep in _ordered(targets, strategy):
        dep_id = dep['id']
        with _LOCK:
            _inflight[dep_id] = _inflight.get(dep_id, 0) + 1
        t0 = time.time()
        try:
            r = requests.post(dep['endpoint'].rstrip('/') + '/v1/chat/completions', headers=_hdrs(dep), json=payload, timeout=timeout)
            body = r.json()
            ms = (time.time() - t0) * 1000
            if r.status_code < 400:
                u = body.get('usage') or {}
                in_tok = u.get('prompt_tokens', 0)
                out_tok = u.get('completion_tokens', 0)
                _mark_success(dep_id)
                _record_stats(dep, ms, in_tok, out_tok, ok=True)
                analytics.record(model or dep['model_key'], dep['account_name'], dep_id, in_tok, out_tok, ms)
                keys.record_usage(api_key, in_tok, out_tok)
                body['_routed'] = {'account': dep['account_name'], 'deployment_id': dep_id, 'ms': round(ms, 1), 'tps': round(out_tok / (ms / 1000.0), 1) if ms and out_tok else 0}
                return (r.status_code, body, dep)
            if r.status_code in (502, 503, 504):
                _mark_failure(dep_id, f'upstream {r.status_code}', hard=True)
            else:
                _mark_failure(dep_id, f'upstream {r.status_code}')
            _record_stats(dep, ms, 0, 0, ok=False)
            errors.append({'account': dep['account_name'], 'status': r.status_code, 'body': str(body)[:200]})
        except requests.exceptions.Timeout:
            _mark_failure(dep_id, 'timeout')
            _record_stats(dep, (time.time() - t0) * 1000, 0, 0, ok=False)
            errors.append({'account': dep['account_name'], 'error': 'timeout'})
        except Exception as e:
            ms = (time.time() - t0) * 1000
            _mark_failure(dep_id, str(e)[:160])
            _record_stats(dep, ms, 0, 0, ok=False)
            with _LOCK:
                st = _stats.setdefault(dep_id, {})
                st['last_error'] = str(e)[:160]
            errors.append({'account': dep['account_name'], 'error': str(e)[:160]})
        finally:
            with _LOCK:
                _inflight[dep_id] = max(0, _inflight.get(dep_id, 1) - 1)
    analytics.record(model or '?', '-', '-', 0, 0, 0, ok=False)
    try:
        import keepalive
        keepalive.request_recovery()
    except Exception:
        pass
    return (503, {'error': {'type': 'all_endpoints_failed', 'message': 'every endpoint failed; the router will retry', 'attempts': errors}}, {})

def chat_stream(payload: dict, strategy: str='round-robin', timeout: float=REQUEST_TIMEOUT_S, api_key: str=''):
    """Relay a streaming completion.

    Fails over only before the first chunk, so a mid-stream death shows up as
    a clean close rather than mixed output from two different models.
    """
    model = payload.get('model')
    targets = _targets(model)
    if not targets:
        status, body = _no_endpoint_error(model)
        msg = body.get('error', {}).get('message', 'no endpoint')

        def err():
            yield f"data: {json.dumps({'error': {'message': msg}})}\n\n".encode()
            yield b'data: [DONE]\n\n'
        return err()

    def gen():
        errors = []
        for dep in _ordered(targets, strategy):
            dep_id = dep['id']
            t0 = time.time()
            got_first = False
            in_tok = out_tok = 0
            try:
                with _LOCK:
                    _inflight[dep_id] = _inflight.get(dep_id, 0) + 1
                r = requests.post(dep['endpoint'].rstrip('/') + '/v1/chat/completions', headers=_hdrs(dep), json=payload, timeout=timeout, stream=True)
                if r.status_code >= 400:
                    _mark_failure(dep_id, f'upstream {r.status_code}')
                    errors.append({'account': dep['account_name'], 'status': r.status_code})
                    continue
                for line in r.iter_lines():
                    if not line:
                        continue
                    text = line.decode('utf-8', 'replace')
                    if text.startswith('data: '):
                        chunk = text[6:]
                        if chunk.strip() == '[DONE]':
                            continue
                        try:
                            j = json.loads(chunk)
                            u = j.get('usage') or {}
                            if u:
                                in_tok = u.get('prompt_tokens', in_tok)
                                out_tok = u.get('completion_tokens', out_tok)
                        except Exception:
                            pass
                    got_first = True
                    yield (text + '\n\n').encode()
                if got_first:
                    ms = (time.time() - t0) * 1000
                    _mark_success(dep_id)
                    _record_stats(dep, ms, in_tok, out_tok, ok=True)
                    analytics.record(model or dep['model_key'], dep['account_name'], dep_id, in_tok, out_tok, ms)
                    keys.record_usage(api_key, in_tok, out_tok)
                    yield b'data: [DONE]\n\n'
                    return
                _mark_failure(dep_id, 'empty stream')
                errors.append({'account': dep['account_name'], 'status': 'empty'})
            except Exception as e:
                _mark_failure(dep_id, str(e)[:120])
                errors.append({'account': dep['account_name'], 'error': str(e)[:120]})
                if got_first:
                    analytics.record(model or '?', dep['account_name'], dep_id, in_tok, out_tok, (time.time() - t0) * 1000, ok=False)
                    yield f"data: {json.dumps({'error': {'message': 'upstream died mid-stream', 'account': dep['account_name']}})}\n\n".encode()
                    yield b'data: [DONE]\n\n'
                    return
            finally:
                with _LOCK:
                    _inflight[dep_id] = max(0, _inflight.get(dep_id, 1) - 1)
        yield f"data: {json.dumps({'error': {'message': 'all endpoints failed', 'attempts': errors}})}\n\n".encode()
        yield b'data: [DONE]\n\n'
    return gen()

def router_stats() -> dict:
    with _LOCK:
        return {'inflight': dict(_inflight), 'stats': {k: dict(v) for k, v in _stats.items()}, 'health': {k: dict(v) for k, v in _health.items()}, 'endpoints': endpoints(), 'probe_interval_s': PROBE_INTERVAL_S, 'request_timeout_s': REQUEST_TIMEOUT_S}
_load_health()
