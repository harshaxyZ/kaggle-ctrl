"""
app.py - Kaggle control dashboard, account pool API, and OpenAI-compatible router.

Run:  python run.py            (serves http://0.0.0.0:8787)

Control API (bearer = a dashboard key from /api/generate-key or api_keys.json):
  GET  /api/state                 everything in one call
  GET  /api/accounts              account pool
  POST /api/accounts              {name, token, username}
  POST /api/accounts/{id}         patch fields
  POST /api/accounts/{id}/delete  remove
  POST /api/accounts/{id}/check   pull live quota for one account
  GET  /api/quota                 pooled + per-account GPU hours
  GET  /api/proxy/quota           $10/day + $100/month spend
  GET  /api/proxy/models          service catalogue
  GET  /api/proxy/session         mint a proxy token
  POST /api/proxy/chat            proxy chat passthrough
  GET  /api/deployments           all serving kernels
  POST /api/server/start          {model_key?, account_id?}
  POST /api/server/stop/{dep_id}  stop one
  POST /api/server/stop           stop all
  POST /api/server/refresh        poll kernels + quotas
  GET  /api/models                self-host candidate models
  GET  /api/router/stats          round-robin counters
  POST /api/generate-key          mint a dashboard key
  GET  /api/keys                  list keys
  POST /api/keys/{key}/revoke     revoke a key

Router (bearer = same dashboard key):
  GET  /v1/models
  POST /v1/chat/completions       round-robin across live deployments
"""
from __future__ import annotations
import os
import threading
import time
from typing import Any
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import accounts
import analytics
import keepalive
import tunnel
import keys
import kaggle_client as kc
import router as rr
import server_manager as sm
app = FastAPI(title='Kaggle Control', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])
OPEN_IF_NO_KEYS = True
LOCAL_HOSTS = {'127.0.0.1', '::1', 'localhost'}

def _extract(authorization: str | None, x_api_key: str | None) -> str:
    if authorization and authorization.lower().startswith('bearer '):
        return authorization[7:].strip()
    return (x_api_key or '').strip()

def _local_trusted(request: Request) -> bool:
    """True only for same-machine calls that are not cross-site browser
    requests. CORS is wide open, so without the Origin check any web page the
    operator has open could drive the dashboard through http://127.0.0.1 -
    browsers send Origin on cross-site fetch, curl/scripts do not."""
    if not (request.client and request.client.host in LOCAL_HOSTS):
        return False
    origin = request.headers.get('origin')
    if origin:
        from urllib.parse import urlparse
        host = urlparse(origin).hostname or ''
        if host not in LOCAL_HOSTS:
            return False
    return True

def require_key(request: Request, authorization: str | None=Header(default=None), x_api_key: str | None=Header(default=None)) -> str:
    """Control API. Requests from the machine running the dashboard are
    trusted; anything from the network needs a valid key. When no key exists
    yet the API stays open so the first key can be minted (open-control
    bootstrap, see OPEN_IF_NO_KEYS)."""
    if _local_trusted(request):
        return ''
    if OPEN_IF_NO_KEYS and (not keys.any_active()):
        return ''
    token = _extract(authorization, x_api_key)
    if not keys.is_valid(token):
        raise HTTPException(status_code=401, detail='invalid or missing dashboard key')
    return token

def require_router_key(request: Request, authorization: str | None=Header(default=None), x_api_key: str | None=Header(default=None)) -> str:
    """Router API. Always needs a valid key, local or not, so a revoked or
    deleted key is rejected everywhere, which is what callers will rely on."""
    token = _extract(authorization, x_api_key)
    if not keys.is_valid(token):
        raise HTTPException(status_code=401, detail='invalid or missing API key')
    return token
max_tokens_default = 256

class StartReq(BaseModel):
    model_config = {'protected_namespaces': ()}
    model_key: str | None = None
    account_id: str | None = None
    accelerator: str = 'NvidiaTeslaT4'

class ChatReq(BaseModel):
    model_config = {'protected_namespaces': ()}
    model: str
    messages: list[dict]
    max_tokens: int = 256

class StopAccountReq(BaseModel):
    account_id: str = ''

class AccountReq(BaseModel):
    name: str
    token: str
    username: str = ''
    notes: str = ''

class AccountPatch(BaseModel):
    model_config = {'protected_namespaces': ()}
    name: str | None = None
    token: str | None = None
    username: str | None = None
    enabled: bool | None = None
    weight: int | None = None
    max_hours: float | None = None
    notes: str | None = None

class KeyReq(BaseModel):
    label: str = 'default'
LOGIN_PAGE = '<!DOCTYPE html>\n<html><head><meta charset="utf-8"/>\n<meta name="viewport" content="width=device-width,initial-scale=1"/>\n<title>kaggle ctrl</title>\n<style>\nbody{background:#0b0d10;color:#e8edf5;font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;\n     display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}\n.box{background:#14171c;border:1px solid #262b34;border-radius:14px;padding:28px;width:340px}\nh1{font-size:16px;margin:0 0 6px}\np{color:#7c8798;font-size:13px;margin:0 0 16px}\ninput{width:100%;box-sizing:border-box;background:#1a1e25;color:#e8edf5;\n      border:1px solid #262b34;border-radius:9px;padding:11px 13px;font:inherit;outline:none}\ninput:focus{border-color:#4f8ef7}\nbutton{width:100%;margin-top:10px;background:#4f8ef7;color:#fff;border:none;border-radius:9px;\n       padding:12px;font:inherit;font-weight:600;cursor:pointer}\n#err{color:#ff8b84;font-size:13px;margin-top:10px;height:16px}\n</style></head><body>\n<div class="box">\n  <h1>kaggle ctrl</h1>\n  <p>This dashboard is private. Enter your API key.</p>\n  <input id="k" type="password" placeholder="sk-kctrl-..." autofocus/>\n  <button onclick="go()">Unlock</button>\n  <div id="err"></div>\n</div>\n<script>\nasync function go(){\n  const k = document.getElementById(\'k\').value.trim();\n  const r = await fetch(\'/api/state\', {headers:{\'Authorization\':\'Bearer \'+k}});\n  if (r.ok) { localStorage.setItem(\'dashkey\', k); location.reload(); }\n  else document.getElementById(\'err\').textContent = \'Invalid key\';\n}\ndocument.getElementById(\'k\').addEventListener(\'keydown\', e => { if (e.key===\'Enter\') go(); });\n</script>\n</body></html>'

@app.get('/', response_class=HTMLResponse)
def index(request: Request, authorization: str | None=Header(default=None), x_api_key: str | None=Header(default=None)) -> str:
    if not _local_trusted(request):
        token = _extract(authorization, x_api_key)
        if not keys.is_valid(token):
            return LOGIN_PAGE
    with open(os.path.join(os.path.dirname(__file__), 'static', 'index.html'), encoding='utf-8') as f:
        return f.read()

@app.get('/health')
def health() -> dict:
    return {'ok': True}
_REFRESH_LOCK = threading.Lock()

def _sync_loop(interval: int=30) -> None:
    while True:
        try:
            if _REFRESH_LOCK.acquire(blocking=False):
                try:
                    sm.refresh()
                    module = __import__('settings').SETTINGS.get('sync_module')
                    if module:
                        __import__(module).sync(list(sm.DEPLOYMENTS.values()))
                finally:
                    _REFRESH_LOCK.release()
        except Exception:
            pass
        time.sleep(interval)

@app.on_event('startup')
def _start_sync() -> None:
    t = threading.Thread(target=_sync_loop, daemon=True, name='kaggle-sync')
    t.start()
    rr.start_prober()
    threading.Thread(target=sm.refresh, daemon=True).start()

def _account_quota() -> dict:
    """GPU hours per account, so each account page can show its own usage."""
    out: dict[str, dict] = {}
    for acct in accounts.load():
        try:
            q = kc.get_accelerator_quota(token=acct.token, config_dir=str(acct.config_dir()))
            if not q.error:
                out[acct.name] = {'used_h': round(q.gpu_used_h, 3), 'total_h': round(q.gpu_total_h, 2), 'remaining_h': round(q.gpu_remaining_h, 3), 'refresh_at': q.gpu_refresh_at}
                continue
        except Exception:
            pass
        out[acct.name] = {'used_h': round(acct.used_hours, 3), 'total_h': acct.max_hours, 'remaining_h': round(max(0.0, acct.max_hours - acct.used_hours), 3), 'refresh_at': acct.reset_at}
    return out

@app.get('/api/state')
def api_state(request: Request, _: str=Depends(require_key)) -> dict[str, Any]:
    request_is_local = bool(request.client and request.client.host in LOCAL_HOSTS)
    quota = kc.get_accelerator_quota()
    proxy_q = kc.get_proxy_quota()
    srv = sm.state()
    accts = [a.as_dict() for a in accounts.load()]
    if srv.get('live_endpoints'):
        live = srv['live_endpoints'][0]
        server_block = dict(live)
        server_block['model_spec'] = live.get('model_spec', {})
        server_block['logs_tail'] = live.get('logs_tail', '')
    else:
        server_block = srv['deployments'][0] if srv['deployments'] else {'status': 'STOPPED', 'endpoint': '', 'api_key': '', 'model_id': '', 'message': 'not running', 'kernel_ref': '', 'uptime_s': 0, 'logs_tail': '', 'model_key': sm.DEFAULT_MODEL, 'model_spec': sm.MODELS[sm.DEFAULT_MODEL].as_dict()}
    return {'ts': __import__('time').time(), 'accelerator': quota.as_dict(), 'proxy_quota': proxy_q.as_dict(), 'server': server_block, 'deployments': srv['deployments'], 'live_endpoints': srv['live_endpoints'], 'pool': srv['pool'], 'accounts': accts, 'models': srv['models'], 'router': rr.router_stats(), 'analytics': analytics.summary(), 'keepalive': keepalive.state(), 'endpoints': rr.endpoints(), 'tunnel': tunnel.status(), 'account_quota': _account_quota(), 'router_keys': [k['key'] for k in keys.all_keys() if k.get('state') == 'active'], 'dashboard_url': f"http://{('127.0.0.1' if request_is_local else 'localhost')}:8787"}

@app.get('/api/accounts')
def api_accounts(_: str=Depends(require_key)) -> list[dict]:
    return [a.as_dict() for a in accounts.load()]

@app.post('/api/accounts')
def api_accounts_add(req: AccountReq, _: str=Depends(require_key)) -> dict:
    acct = accounts.add(req.name, req.token, req.username, req.notes)
    return {'ok': True, 'account': acct.as_dict()}

@app.post('/api/accounts/{account_id}')
def api_accounts_patch(account_id: str, req: AccountPatch, _: str=Depends(require_key)) -> dict:
    acct = accounts.update(account_id, **req.model_dump(exclude_none=True))
    if not acct:
        raise HTTPException(status_code=404, detail='no such account')
    return {'ok': True, 'account': acct.as_dict()}

@app.post('/api/accounts/{account_id}/delete')
def api_accounts_delete(account_id: str, _: str=Depends(require_key)) -> dict:
    return {'ok': accounts.remove(account_id)}

@app.post('/api/accounts/{account_id}/check')
def api_accounts_check(account_id: str, _: str=Depends(require_key)) -> dict:
    acct = accounts.get(account_id)
    if not acct:
        raise HTTPException(status_code=404, detail='no such account')
    q = kc.get_accelerator_quota(token=acct.token, config_dir=str(acct.config_dir()))
    if not q.error:
        accounts.refresh_quota(account_id, q.gpu_used_h, q.gpu_total_h, q.gpu_refresh_at)
    return {'ok': not q.error, 'quota': q.as_dict()}

@app.get('/api/quota')
def api_quota(_: str=Depends(require_key)) -> dict:
    per = []
    for a in accounts.load():
        q = kc.get_accelerator_quota(token=a.token, config_dir=str(a.config_dir()))
        per.append({'account': a.name, 'account_id': a.id, **q.as_dict()})
    pool = accounts.pool_summary()
    return {'pool': pool, 'accounts': per}

@app.get('/api/proxy/quota')
def api_proxy_quota(_: str=Depends(require_key)) -> dict:
    return kc.get_proxy_quota().as_dict()

@app.get('/api/proxy/models')
def api_proxy_models(_: str=Depends(require_key)) -> list[dict]:
    return [m.as_dict() for m in kc.get_proxy_catalogue()]

@app.get('/api/proxy/session')
def api_proxy_session(_: str=Depends(require_key)) -> dict:
    try:
        s = kc.create_proxy_session()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f'proxy token failed: {e}')
    return {'base_url': s.openai_base + '/v1', 'api_key': s.token, 'raw_base_uri': s.base_uri, 'expiry': s.expiry, 'available_models': kc.proxy_models_list(s)}

@app.post('/api/proxy/chat')
def api_proxy_chat(req: ChatReq, _: str=Depends(require_key)) -> dict:
    try:
        s = kc.create_proxy_session()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f'proxy token failed: {e}')
    return kc.proxy_chat(s, req.model, req.messages, req.max_tokens)

@app.get('/api/models')
def api_models(_: str=Depends(require_key)) -> list[dict]:
    return [m.as_dict() for m in sm.MODELS.values()]

@app.get('/api/deployments')
def api_deployments(_: str=Depends(require_key)) -> list[dict]:
    return sm.state()['deployments']

@app.get('/api/server')
def api_server(_: str=Depends(require_key)) -> dict:
    return sm.state()

@app.post('/api/server/start')
def api_server_start(req: StartReq, _: str=Depends(require_key)) -> dict:
    if req.model_key and req.model_key not in sm.MODELS:
        raise HTTPException(status_code=400, detail=f'unknown model_key {req.model_key}')
    return sm.start(model_key=req.model_key, account_id=req.account_id, accelerator=req.accelerator)

@app.post('/api/server/stop/{dep_id}')
def api_server_stop_one(dep_id: str, _: str=Depends(require_key)) -> dict:
    return sm.stop(dep_id)

@app.post('/api/server/stop')
def api_server_stop(_: str=Depends(require_key)) -> dict:
    return sm.stop(None)

@app.post('/api/server/stop-by-account')
def api_server_stop_account(req: StopAccountReq | None=None, _: str=Depends(require_key)) -> dict:
    """Stop the deployment belonging to one account, leaving the other alone."""
    acct_id = (req.account_id if req else '') or ''
    if not acct_id:
        raise HTTPException(status_code=400, detail='account_id required')
    stopped = []
    for d in list(sm.DEPLOYMENTS.values()):
        if d.account_id != acct_id:
            continue
        res = sm.stop(d.id)
        stopped.append({'deployment_id': d.id, 'account': d.account_name, 'result': res})
    if not stopped:
        return {'ok': True, 'message': 'nothing running for that account', 'stopped': []}
    return {'ok': True, 'message': 'stopped', 'stopped': stopped}

@app.post('/api/server/refresh')
def api_server_refresh(_: str=Depends(require_key)) -> dict:
    return sm.refresh()

@app.get('/api/router/stats')
def api_router_stats(_: str=Depends(require_key)) -> dict:
    return rr.router_stats()

@app.get('/api/analytics')
def api_analytics(window: int=0, _: str=Depends(require_key)) -> dict:
    return analytics.summary(window_s=window or None)

@app.post('/api/analytics/clear')
def api_analytics_clear(_: str=Depends(require_key)) -> dict:
    analytics.clear()
    return {'ok': True}

@app.get('/api/keepalive')
def api_keepalive_get(_: str=Depends(require_key)) -> dict:
    return keepalive.state()

@app.post('/api/keepalive/on')
def api_keepalive_on(req: KeyReq | None=None, _: str=Depends(require_key)) -> dict:
    return keepalive.start()

@app.post('/api/keepalive/off')
def api_keepalive_off(_: str=Depends(require_key)) -> dict:
    return keepalive.stop()

@app.get('/api/simple')
def api_simple(_: str=Depends(require_key)) -> dict:
    """One small payload for the simplified dashboard."""
    srv = sm.state()
    deps = srv['deployments']
    live = [d for d in deps if d['status'] == 'RUNNING' and d['endpoint']]
    primary = live[0] if live else deps[-1] if deps else None
    gpu = kc.get_accelerator_quota()
    proxy = kc.get_proxy_quota()
    pool = dict(srv['pool'])
    pool['proxy_daily_usd'] = round((proxy.as_dict().get('daily') or {}).get('remaining', 0), 2)
    pool['proxy_monthly_usd'] = round((proxy.as_dict().get('monthly') or {}).get('remaining', 0), 2)
    return {'ts': __import__('time').time(), 'server': {'status': primary['status'] if primary else 'STOPPED', 'model_key': primary['model_key'] if primary else sm.DEFAULT_MODEL, 'model_label': primary['model_spec']['label'] if primary else '', 'uptime_s': primary['uptime_s'] if primary else 0, 'endpoint': primary['endpoint'] if primary and primary['status'] == 'RUNNING' else '', 'message': primary['message'] if primary else ''}, 'deployments': [{'id': d['id'], 'account': d['account_name'], 'status': d['status'], 'model': d['model_key'], 'endpoint': d['endpoint'], 'uptime_s': d['uptime_s'], 'message': d['message']} for d in deps if d['status'] != 'STOPPED'], 'router': {'base_url': '/v1', 'live': len(live)}, 'pool': pool, 'gpu': gpu.as_dict()['gpu'], 'proxy': proxy.as_dict(), 'keepalive': keepalive.state(), 'models': [m.as_dict() for m in sm.MODELS.values()], 'accounts': [a.as_dict() for a in accounts.load()], 'analytics': analytics.summary()}

class CmdReq(BaseModel):
    command: str

@app.post('/api/cmd')
def api_cmd(req: CmdReq, _: str=Depends(require_key)) -> dict:
    """Run one dashboard command from any agent with HTTP access.

    Body: {"command": "start configured-model"} or {"command": "stop"}, etc.
    """
    parts = (req.command or '').split()
    if not parts:
        return {'ok': False, 'error': 'empty command'}
    cmd, args = (parts[0].lower(), parts[1:])
    if cmd == 'start':
        return {'ok': True, 'action': 'start', **sm.start(model_key=args[0] if args else None)}
    if cmd == 'stop':
        return {'ok': True, 'action': 'stop', **sm.stop(None)}
    if cmd in ('status', 'state'):
        return {'ok': True, 'action': 'status', **sm.state()}
    if cmd == 'endpoint':
        live = sm.state()['live_endpoints']
        return {'ok': True, 'action': 'endpoint', 'endpoints': [{'base_url': d['endpoint'] + '/v1', 'api_key': d['api_key'], 'account': d['account_name'], 'model': d['model_id']} for d in live]}
    if cmd == 'models':
        return {'ok': True, 'action': 'models', 'selfhost': [m.as_dict() for m in sm.MODELS.values()]}
    if cmd in ('proxy-models', 'proxymodels'):
        return {'ok': True, 'action': 'proxy-models', 'models': [m.as_dict() for m in kc.get_proxy_catalogue()]}
    if cmd == 'mint':
        try:
            s = kc.create_proxy_session()
        except Exception as e:
            return {'ok': False, 'error': str(e)}
        return {'ok': True, 'action': 'mint', 'base_url': s.openai_base + '/v1', 'api_key': s.token, 'expiry': s.expiry}
    if cmd == 'poll':
        return {'ok': True, 'action': 'poll', **sm.refresh()}
    if cmd == 'quota':
        return {'ok': True, 'action': 'quota', 'quota': kc.get_accelerator_quota().as_dict(), 'proxy': kc.get_proxy_quota().as_dict()}
    if cmd in ('accounts', 'pool'):
        return {'ok': True, 'action': 'accounts', 'accounts': [a.as_dict() for a in accounts.load()], 'pool': accounts.pool_summary()}
    if cmd == 'help':
        return {'ok': True, 'commands': ['start [model]', 'stop', 'status', 'endpoint', 'models', 'proxy-models', 'mint', 'poll', 'quota', 'accounts']}
    return {'ok': False, 'error': f'unknown command {cmd!r}'}

class NewKeyReq(BaseModel):
    name: str = ''
    expires_days: float | None = None

class RenameReq(BaseModel):
    name: str

@app.post('/api/generate-key')
def api_generate_key(req: NewKeyReq | None=None, _: str=Depends(require_key)) -> dict:
    row = keys.create(req.name if req else '', req.expires_days if req else None)
    return {'ok': True, 'key': row['key'], 'name': row['name'], 'expires_at': row['expires_at']}

@app.post('/api/keys')
def api_keys_create(req: NewKeyReq, _: str=Depends(require_key)) -> dict:
    row = keys.create(req.name, req.expires_days)
    return {'ok': True, **row}

def _mask(k: str) -> str:
    if len(k) <= 16:
        return k[:4] + '...' + k[-4:]
    return k[:12] + '...' + k[-6:]

@app.get('/api/keys')
def api_keys(reveal: bool=False, _: str=Depends(require_key)) -> list[dict]:
    rows = keys.all_keys()
    for r in rows:
        r['key_masked'] = _mask(r['key'])
        if not reveal:
            r['key'] = r['key_masked']
    return rows

@app.post('/api/keys/delete-all')
def api_keys_delete_all(_: str=Depends(require_key)) -> dict:
    """Remove every key. Used to reset the dashboard to a clean state."""
    n = keys.delete_all()
    return {'ok': True, 'deleted': n}

@app.post('/api/keys/{key}/revoke')
def api_keys_revoke(key: str, _: str=Depends(require_key)) -> dict:
    return {'ok': keys.revoke(key)}

@app.post('/api/keys/{key}/unrevoke')
def api_keys_unrevoke(key: str, _: str=Depends(require_key)) -> dict:
    return {'ok': keys.unrevoke(key)}

@app.delete('/api/keys/{key}')
@app.post('/api/keys/{key}/delete')
def api_keys_delete(key: str, _: str=Depends(require_key)) -> dict:
    return {'ok': keys.delete(key)}

@app.post('/api/keys/{key}/rename')
def api_keys_rename(key: str, req: RenameReq, _: str=Depends(require_key)) -> dict:
    return {'ok': keys.rename(key, req.name)}

@app.get('/api/tunnel')
def api_tunnel_get(_: str=Depends(require_key)) -> dict:
    """Status of the public URL for this dashboard."""
    return tunnel.status()

@app.post('/api/tunnel/start')
def api_tunnel_start(_: str=Depends(require_key)) -> dict:
    """Expose the dashboard publicly so a remote bot can reach it.

    Only needed when the bot does not run on this machine. The URL is
    published to the ntfy topic "per-install discovery topic" and to dashboard_url.txt,
    so a remote client can always discover the current one.
    """
    return tunnel.start(port=int(os.environ.get('PORT', '8787')))

@app.post('/api/tunnel/stop')
def api_tunnel_stop(_: str=Depends(require_key)) -> dict:
    return tunnel.stop()

@app.get('/v1/models')
def v1_models(_: str=Depends(require_router_key)) -> dict:
    return rr.models()

@app.post('/v1/chat/completions')
def v1_chat(payload: dict, _: str=Depends(require_router_key)):
    """OpenAI-compatible chat.

    The response carries X-Kaggle-Account and X-Kaggle-Deployment headers so a
    client can see which account's endpoint actually served the request.
    """
    strategy = payload.pop('_strategy', 'round-robin')
    if payload.get('stream'):
        return StreamingResponse(rr.chat_stream(payload, strategy=strategy, api_key=_), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    payload.pop('stream', None)
    status, body, served = rr.chat(payload, strategy=strategy, api_key=_)
    headers = {}
    if served:
        headers['X-Kaggle-Account'] = served.get('account_name', '')
        headers['X-Kaggle-Deployment'] = served.get('id', '')
        headers['X-Kaggle-Model'] = served.get('model_id', '')
    return JSONResponse(status_code=status, content=body, headers=headers)

@app.get('/api/endpoints')
def api_endpoints(_: str=Depends(require_key)) -> list[dict]:
    """One row per live deployment: account, base_url, api_key, health, tps."""
    if _REFRESH_LOCK.acquire(blocking=False):
        try:
            sm.refresh()
        except Exception:
            pass
        finally:
            _REFRESH_LOCK.release()
    return rr.endpoints()
