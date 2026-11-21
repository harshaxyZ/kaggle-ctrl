"""Standalone Anthropic Messages API proxy for compatible clients.

Listens on 127.0.0.1:8788, converts Anthropic /v1/messages to the gateway's
OpenAI /v1/chat/completions (127.0.0.1:8787), and converts responses back.
Exposed publicly through its own cloudflared quick tunnel so the dashboard
tunnel and process stay untouched.

Auth: the caller's Bearer token / x-api-key is forwarded to the gateway, which
enforces validity. No keys are stored here.
"""
from __future__ import annotations
import json
import re
import subprocess
import threading
import time
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import anthropic_adapter as ad
GATEWAY = 'http://127.0.0.1:8787'
PORT = 8788
app = FastAPI(title='kaggle-ctrl anthropic proxy')

def _auth_headers(request: Request) -> dict:
    h = {}
    auth = request.headers.get('authorization')
    xkey = request.headers.get('x-api-key')
    if auth:
        h['Authorization'] = auth
    if xkey:
        h['x-api-key'] = xkey
    return h

def _error(status: int, etype: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={'type': 'error', 'error': {'type': etype, 'message': message}})

@app.post('/v1/messages')
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _error(400, 'invalid_request_error', 'request body must be JSON')
    requested_model = str(body.get('model') or ad.TARGET_MODEL)
    payload = ad.to_openai(body)
    headers = {'Content-Type': 'application/json', **_auth_headers(request)}
    if payload.get('stream'):

        def upstream_lines():
            try:
                with requests.post(GATEWAY + '/v1/chat/completions', headers=headers, json=payload, timeout=600, stream=True) as r:
                    if r.status_code >= 400:
                        yield f"data: {json.dumps({'error': {'message': 'gateway %s: %s' % (r.status_code, r.text[:200])}})}".encode()
                        return
                    for line in r.iter_lines():
                        if line:
                            yield line
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': 'upstream unreachable: %s' % str(e)[:160]}})}".encode()
        return StreamingResponse(ad.stream_anthropic(upstream_lines(), requested_model), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    payload.pop('stream', None)
    try:
        r = requests.post(GATEWAY + '/v1/chat/completions', headers=headers, json=payload, timeout=600)
    except Exception as e:
        return _error(503, 'api_error', f'upstream unreachable: {str(e)[:160]}')
    if r.status_code == 401:
        return _error(401, 'authentication_error', 'invalid API key')
    if r.status_code >= 400:
        return _error(r.status_code, 'api_error', f'gateway error: {r.text[:200]}')
    try:
        body_out = r.json()
    except Exception:
        return _error(502, 'api_error', 'gateway returned non-JSON')
    return JSONResponse(ad.from_openai(body_out, requested_model))

@app.post('/v1/messages/count_tokens')
async def count_tokens(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _error(400, 'invalid_request_error', 'request body must be JSON')
    return {'input_tokens': ad.count_tokens(body)}

@app.post('/v1/chat/completions')
async def openai_chat(request: Request):
    """OpenAI-shaped passthrough to the gateway, so the single public URL
    serves both API shapes (Anthropic /v1/messages + OpenAI /v1/chat/completions)."""
    try:
        body = await request.json()
    except Exception:
        return _error(400, 'invalid_request_error', 'request body must be JSON')
    headers = {'Content-Type': 'application/json', **_auth_headers(request)}
    if body.get('stream'):

        def gen():
            try:
                with requests.post(GATEWAY + '/v1/chat/completions', headers=headers, json=body, timeout=600, stream=True) as r:
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': 'upstream unreachable: %s' % str(e)[:160]}})}\n\n".encode()
        return StreamingResponse(gen(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    body.pop('stream', None)
    try:
        r = requests.post(GATEWAY + '/v1/chat/completions', headers=headers, json=body, timeout=600)
    except Exception as e:
        return _error(503, 'api_error', f'upstream unreachable: {str(e)[:160]}')
    try:
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception:
        return _error(502, 'api_error', f'gateway error: {r.text[:200]}')

@app.get('/v1/models')
async def models():
    return {'object': 'list', 'data': [{'id': ad.TARGET_MODEL, 'object': 'model', 'created': 0, 'owned_by': 'kaggle-ctrl'}], 'has_more': False, 'first_id': ad.TARGET_MODEL, 'last_id': ad.TARGET_MODEL}

@app.get('/health')
async def health():
    return {'ok': True, 'target': ad.TARGET_MODEL}

def start_tunnel() -> str:
    proc = subprocess.Popen(['bin/cloudflared', 'tunnel', '--url', f'http://127.0.0.1:{PORT}', '--no-autoupdate'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = ''
    deadline = time.time() + 60
    for line in proc.stdout:
        m = re.search('https://[a-z0-9\\-]+\\.trycloudflare\\.com', line)
        if m:
            url = m.group(0)
            break
        if time.time() > deadline:
            break

    def drain():
        try:
            for _ in proc.stdout:
                pass
        except Exception:
            pass
    threading.Thread(target=drain, daemon=True).start()
    if url:
        with open('anthropic_url.txt', 'w') as f:
            f.write(url)
    return url
if __name__ == '__main__':
    public = start_tunnel()
    print(f'ANTHROPIC PUBLIC URL: {public}', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='info')
