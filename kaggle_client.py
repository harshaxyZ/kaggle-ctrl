"""
kaggle_client.py — thin, dependency-light client for the Kaggle API surface we need.

Covers:
  * accelerator quota (GPU hours / TPU hours, weekly reset)
  * Model Proxy quota ($10/day, $100/month) + catalogue + OpenAI-compatible chat
  * kernel lifecycle (push / status / output / delete) for the serving notebook

Auth: KGAT_... token in KAGGLE_API_TOKEN, or ~/.kaggle/access_token.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import requests
KAGGLE_API = 'https://www.kaggle.com/api/v1'
KG_CLI = 'kaggle'
USER_AGENT = 'kaggle-cli/2.2.5 (kaggle-dashboard)'

def get_token(token: str | None=None) -> str:
    if token:
        return token.strip()
    tok = os.environ.get('KAGGLE_API_TOKEN', '').strip()
    if tok:
        return tok
    p = Path.home() / '.kaggle' / 'access_token'
    if p.exists():
        return p.read_text(encoding='utf-8').strip()
    raise RuntimeError('No Kaggle token found. Set KAGGLE_API_TOKEN or write it to ~/.kaggle/access_token')

def _hdrs(token: str | None=None) -> dict:
    return {'Authorization': f'Bearer {get_token(token)}', 'Content-Type': 'application/json', 'User-Agent': USER_AGENT, 'Accept': 'application/json'}

@dataclass
class AcceleratorQuota:
    gpu_used_h: float = 0.0
    gpu_total_h: float = 0.0
    gpu_refresh_at: str = ''
    tpu_used_h: float = 0.0
    tpu_total_h: float = 0.0
    tpu_refresh_at: str = ''
    error: str = ''

    @property
    def gpu_remaining_h(self) -> float:
        return max(0.0, self.gpu_total_h - self.gpu_used_h)

    @property
    def tpu_remaining_h(self) -> float:
        return max(0.0, self.tpu_total_h - self.tpu_used_h)

    @property
    def gpu_used_pct(self) -> float:
        return self.gpu_used_h / self.gpu_total_h * 100 if self.gpu_total_h else 0.0

    @property
    def gpu_remaining_pct(self) -> float:
        return self.gpu_remaining_h / self.gpu_total_h * 100 if self.gpu_total_h else 0.0

    def as_dict(self) -> dict:
        return {'gpu': {'used_h': round(self.gpu_used_h, 3), 'total_h': round(self.gpu_total_h, 3), 'remaining_h': round(self.gpu_remaining_h, 3), 'used_pct': round(self.gpu_used_pct, 1), 'remaining_pct': round(self.gpu_remaining_pct, 1), 'refresh_at': self.gpu_refresh_at}, 'tpu': {'used_h': round(self.tpu_used_h, 3), 'total_h': round(self.tpu_total_h, 3), 'remaining_h': round(self.tpu_remaining_h, 3), 'refresh_at': self.tpu_refresh_at}, 'error': self.error}

def _iso(t: str) -> str:
    return t

def get_accelerator_quota(token: str | None=None, config_dir: str | None=None) -> AcceleratorQuota:
    """GET accelerator quota statistics (GPU/TPU hours + weekly reset)."""
    q = AcceleratorQuota()
    legacy = bool(token) and (not token.startswith('KGAT_'))
    if not legacy:
        try:
            r = requests.get(f'{KAGGLE_API}/kernels/quota', headers=_hdrs(token), timeout=30)
            if r.status_code == 200:
                j = r.json()
                gpu = j.get('gpuQuota') or {}
                tpu = j.get('tpuQuota') or {}

                def hours(d: dict, key: str) -> float:
                    v = d.get(key) or {}
                    return (v.get('seconds', 0) or 0) / 3600.0
                q.gpu_used_h = hours(gpu, 'timeUsed')
                q.gpu_total_h = hours(gpu, 'totalTimeAllowed')
                q.gpu_refresh_at = j.get('quotaRefreshTime') or ''
                q.tpu_used_h = hours(tpu, 'timeUsed')
                q.tpu_total_h = hours(tpu, 'totalTimeAllowed')
                q.tpu_refresh_at = q.gpu_refresh_at
                return q
            q.error = f'quota http {r.status_code}'
        except Exception as e:
            q.error = f'quota error: {e}'
    try:
        out = _cli(['quota'], timeout=60, config_dir=config_dir, token=token)
        parsed = False
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] in ('GPU', 'TPU'):
                res, used, rem, total = (parts[0], parts[1], parts[2], parts[3])
                used_h = float(used.rstrip('h'))
                rem_h = float(rem.rstrip('h'))
                total_h = float(total.rstrip('h'))
                refresh = parts[4]
                if res == 'GPU':
                    q.gpu_used_h = used_h
                    q.gpu_total_h = total_h
                    q.gpu_refresh_at = refresh
                else:
                    q.tpu_used_h = used_h
                    q.tpu_total_h = total_h
                    q.tpu_refresh_at = refresh
                parsed = True
        if parsed:
            q.error = ''
    except Exception as e:
        if not q.error:
            q.error = f'cli quota error: {e}'
    return q

@dataclass
class ProxyQuota:
    daily_used: float = 0.0
    daily_total: float = 0.0
    monthly_used: float = 0.0
    monthly_total: float = 0.0
    error: str = ''

    @property
    def daily_remaining(self) -> float:
        return max(0.0, self.daily_total - self.daily_used)

    @property
    def monthly_remaining(self) -> float:
        return max(0.0, self.monthly_total - self.monthly_used)

    def as_dict(self) -> dict:
        return {'daily': {'used': round(self.daily_used, 4), 'total': round(self.daily_total, 2), 'remaining': round(self.daily_remaining, 4)}, 'monthly': {'used': round(self.monthly_used, 4), 'total': round(self.monthly_total, 2), 'remaining': round(self.monthly_remaining, 4)}, 'error': self.error}

def get_proxy_quota(token: str | None=None) -> ProxyQuota:
    """$10/day and $100/month Model Proxy spend quota."""
    q = ProxyQuota()
    try:
        r = requests.get(f'{KAGGLE_API}/models/proxy/quota', headers=_hdrs(token), timeout=30)
        if r.status_code == 200:
            j = r.json()
            balances = j if isinstance(j, list) else j.get('quotaBalances') or j.get('quota_balances') or []
            for b in balances:
                period = str(b.get('refillPeriod') or b.get('refill_period') or '').lower()
                used = float(b.get('quotaUsed') or b.get('quota_used') or 0)
                total = float(b.get('totalQuotaAllowed') or b.get('total_quota_allowed') or 0)
                if 'daily' in period:
                    q.daily_used, q.daily_total = (used, total)
                elif 'monthly' in period:
                    q.monthly_used, q.monthly_total = (used, total)
            return q
        q.error = f'proxy quota http {r.status_code}'
    except Exception as e:
        q.error = f'proxy quota error: {e}'
    try:
        out = _cli(['benchmarks', 'quota'], timeout=60)
        parsed = False
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 4 and p[0] in ('Daily', 'Monthly'):
                used = float(p[1].lstrip('$'))
                rem = float(p[2].lstrip('$'))
                total = float(p[3].lstrip('$'))
                if p[0] == 'Daily':
                    q.daily_used, q.daily_total = (total - rem, total)
                else:
                    q.monthly_used, q.monthly_total = (total - rem, total)
                parsed = True
        if parsed:
            q.error = ''
    except Exception as e:
        if not q.error:
            q.error = f'cli proxy quota error: {e}'
    return q

@dataclass
class ProxyModel:
    slug: str
    display_name: str
    org: str
    reasoning: bool = False
    vision: bool = False
    open_weights: bool = False
    tier: str = 'closed'
    note: str = ''
    available: bool = False

    def as_dict(self) -> dict:
        return {'slug': self.slug, 'display_name': self.display_name, 'org': self.org, 'reasoning': self.reasoning, 'vision': self.vision, 'open_weights': self.open_weights, 'tier': self.tier, 'note': self.note, 'available': self.available}
LIVE_PROXY_MODELS = set(__import__('settings').SETTINGS.get('live_proxy_models', []))
PROXY_MODEL_META = __import__('settings').SETTINGS.get('proxy_model_meta', {})

def get_proxy_catalogue(api_token: str | None=None) -> list[ProxyModel]:
    """Model catalogue returned by the configured service."""
    models: list[ProxyModel] = []
    token = None
    for _ in range(8):
        url = f'{KAGGLE_API}/benchmarks/models'
        if token:
            url += '?pageToken=' + token
        try:
            r = requests.get(url, headers=_hdrs(api_token), timeout=30)
            if r.status_code != 200:
                break
            j = r.json()
        except Exception:
            break
        arr = j.get('benchmarkModels') or []
        for m in arr:
            ver = m.get('version') or {}
            slug = ver.get('modelProxySlug') or m.get('slug') or ''
            if not slug:
                continue
            meta = PROXY_MODEL_META.get(slug, {})
            models.append(ProxyModel(slug=slug, display_name=m.get('displayName') or slug, org=(ver.get('organization') or {}).get('name') or '?', reasoning=bool(ver.get('supportsReasoningEffort')) or meta.get('reasoning', False), vision='vl' in slug.lower() or 'vision' in slug.lower(), open_weights=meta.get('open_weights', False), tier=meta.get('tier', 'closed'), note=meta.get('note', ''), available=not LIVE_PROXY_MODELS or slug in LIVE_PROXY_MODELS))
        token = j.get('nextPageTokenNullable')
        if not token or not arr:
            break
    seen: dict[str, ProxyModel] = {}
    for m in models:
        seen.setdefault(m.slug, m)
    out = list(seen.values())

    def rank(m: ProxyModel) -> tuple:
        return (0 if m.available else 1, 0 if m.open_weights else 1, m.display_name.lower())
    return sorted(out, key=rank)

@dataclass
class ProxySession:
    token: str = ''
    base_uri: str = ''
    expiry: str = ''

    @property
    def openai_base(self) -> str:
        return self.base_uri.rstrip('/') + '/openapi' if self.base_uri else ''

    @property
    def active(self) -> bool:
        return bool(self.token and self.base_uri)

def create_proxy_session(token: str | None=None) -> ProxySession:
    h = {'Authorization': f'Bearer {get_token(token)}', 'Content-Type': 'application/json', 'User-Agent': USER_AGENT, 'Accept': 'application/json'}
    r = requests.post(f'{KAGGLE_API}/models/proxy/token', headers=h, data='{}', timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f'proxy token http {r.status_code}: {r.text[:200]}')
    j = r.json()
    return ProxySession(token=j.get('token', ''), base_uri=j.get('baseUri', ''), expiry=j.get('expiryTime', ''))

def proxy_chat(session: ProxySession, model: str, messages: list[dict], max_tokens: int=256) -> dict:
    """Call the proxy through its OpenAI-compatible endpoint. Returns raw JSON."""
    url = session.openai_base + '/v1/chat/completions'
    h = {'Authorization': f'Bearer {session.token}', 'Content-Type': 'application/json', 'User-Agent': USER_AGENT, 'Accept': 'application/json'}
    resp = requests.post(url, headers=h, json={'model': model, 'messages': messages, 'max_tokens': max_tokens}, timeout=120)
    try:
        return resp.json()
    except Exception:
        return {'error': {'message': resp.text[:300]}}

def proxy_models_list(session: ProxySession) -> list[str]:
    url = session.openai_base + '/v1/models'
    h = {'Authorization': f'Bearer {session.token}', 'User-Agent': USER_AGENT}
    try:
        r = requests.get(url, headers=h, timeout=30)
        j = r.json()
        return [m.get('id') for m in j.get('data') or [] if m.get('id')]
    except Exception:
        return []

@dataclass
class KernelInfo:
    ref: str
    status: str = 'UNKNOWN'
    endpoint: str = ''
    api_key: str = ''
    model: str = ''
    logs: str = ''
    last_error: str = ''
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {'ref': self.ref, 'status': self.status, 'endpoint': self.endpoint, 'api_key': self.api_key, 'model': self.model, 'last_error': self.last_error, 'logs_tail': self.logs[-4000:] if self.logs else ''}

def _kaggle_cmd() -> list[str]:
    """Resolve how to invoke the Kaggle CLI.

    The CLI is a console script that lives next to whichever Python
    interpreter has the `kaggle` package installed. Shelling out to a bare
    "kaggle" works only if that interpreter's Scripts directory happens to be
    on PATH, and it is easy to end up running the dashboard under a different
    Python than the one that owns the CLI - which makes every CLI call fail
    with "No module named kaggle" and looks like Kaggle itself is broken.

    So: try the PATH console script, then any interpreter that actually has
    the package, then the current interpreter as "python -m kaggle".
    """
    import shutil
    import sys
    exe = shutil.which(KG_CLI)
    if exe:
        return [exe]
    for py in _pythons_with_kaggle():
        return [py, '-m', 'kaggle']
    return [sys.executable, '-m', 'kaggle']

def _pythons_with_kaggle() -> list[str]:
    """Every python on this box that can import kaggle, newest first."""
    import glob
    import os
    import subprocess
    import sys
    candidates: list[str] = []
    roots = [os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Python'), 'C:\\python']
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for ver in sorted(glob.glob(os.path.join(root, 'Python3*')), reverse=True):
            exe = os.path.join(ver, 'python.exe')
            if os.path.exists(exe):
                candidates.append(exe)
    if sys.executable:
        candidates.append(sys.executable)
    good: list[str] = []
    for exe in candidates:
        try:
            r = subprocess.run([exe, '-c', 'import kaggle'], capture_output=True, timeout=30)
            if r.returncode == 0:
                good.append(exe)
        except Exception:
            continue
    return good

def _cli(args: list[str], timeout: int=120, cwd: str | None=None, config_dir: str | None=None, token: str | None=None) -> str:
    """Run the kaggle CLI, optionally as a specific account.

    The CLI resolves the token from KAGGLE_API_TOKEN or ~/.kaggle/access_token
    before it ever looks at KAGGLE_CONFIG_DIR, so pointing only the config dir
    at another account silently keeps using the default one. Both have to be
    set, and any inherited KAGGLE_* vars have to be cleared so they cannot win.
    """
    cmd = _kaggle_cmd() + args
    env = {k: v for k, v in os.environ.items() if not k.startswith('KAGGLE_')}
    env['PYTHONUTF8'] = '1'
    if token and token.startswith('KGAT_'):
        env['KAGGLE_API_TOKEN'] = token
    elif token:
        pass
    elif os.environ.get('KAGGLE_API_TOKEN', '').startswith('KGAT_'):
        env['KAGGLE_API_TOKEN'] = os.environ['KAGGLE_API_TOKEN']
    if config_dir:
        env['KAGGLE_CONFIG_DIR'] = config_dir
        if token and (not token.startswith('KGAT_')):
            home = Path(tempfile.mkdtemp(prefix='kcli-home-'))
            kd = home / '.kaggle'
            kd.mkdir()
            src = Path(config_dir) / 'kaggle.json'
            if src.exists():
                shutil.copy2(src, kd / 'kaggle.json')
                (kd / 'kaggle.json').chmod(384)
            env['HOME'] = str(home)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or 'kaggle CLI failed')[:400])
    return proc.stdout or ''

def list_kernels(account_username: str | None=None, config_dir: str | None=None, token: str | None=None) -> list[str]:
    """List the kernel refs ("user/slug") owned by an account.

    The CLI prints a table whose first column is the ref. We parse that rather
    than hitting the REST API because the CLI already resolves the per-account
    config dir and token correctly.
    """
    try:
        out = _cli(['kernels', 'list', '--mine'], timeout=120, config_dir=config_dir, token=token)
    except Exception:
        return []
    refs: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        ref = parts[0]
        if '/' not in ref:
            continue
        if account_username and (not ref.startswith(account_username + '/')):
            continue
        refs.append(ref)
    return refs

def kernel_status(ref: str, config_dir: str | None=None, token: str | None=None) -> str:
    try:
        out = _cli(['kernels', 'status', ref], timeout=60, config_dir=config_dir, token=token)
    except Exception as e:
        return f'ERROR: {e}'
    m = re.search('status "([^"]+)"', out)
    return m.group(1) if m else out.strip()[:120]
ENDPOINT_RE = re.compile('https://[a-z0-9\\-]+\\.(?:trycloudflare\\.com|shares\\.zrok\\.io|loca\\.lt)')
APIKEY_RE = re.compile('\\bsk-[A-Za-z0-9_\\-]{12,}')
MODEL_RE = re.compile('^\\s*MODEL\\s*[:=]\\s*(\\S+)', re.MULTILINE)

def kernel_output(ref: str, workdir: str, config_dir: str | None=None, token: str | None=None) -> str:
    """Download the kernel's logs into workdir, return the log text."""
    d = Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob('*.log'):
        f.unlink()
    try:
        _cli(['kernels', 'output', ref], timeout=300, cwd=str(d), config_dir=config_dir, token=token)
    except Exception as e:
        return f'(output fetch failed: {e})'
    logs = sorted(d.glob('*.log'))
    if not logs:
        return ''
    return logs[0].read_text(encoding='utf-8', errors='replace')

def kernel_delete(ref: str, config_dir: str | None=None, token: str | None=None) -> str:
    try:
        _cli(['kernels', 'delete', ref, '-y'], timeout=120, config_dir=config_dir, token=token)
        return 'deleted'
    except Exception as e:
        return f'delete failed: {e}'

def kernel_push(kernel_dir: str, accelerator: str='NvidiaTeslaT4', timeout: int=300, config_dir: str | None=None, token: str | None=None) -> str:
    """Push and run a kernel.

    Do not pass -t/--timeout here: it is the kernel's *run* limit, not the
    upload limit. Passing it cancelled every serving kernel a few minutes
    after it went live, which looked exactly like a crash.
    """
    try:
        out = _cli(['kernels', 'push', '-p', kernel_dir], timeout=timeout + 120, config_dir=config_dir, token=token)
        return out.strip()
    except Exception as e:
        return f'push failed: {e}'

def parse_serving_log(log_text: str) -> dict:
    """Pull ENDPOINT / API KEY / MODEL out of the serving notebook's stdout."""
    endpoint = ''
    api_key = ''
    model = ''
    m = ENDPOINT_RE.search(log_text)
    if m:
        endpoint = m.group(0)
    km = re.search('API\\s*KEY\\s*[:=]\\s*(' + APIKEY_RE.pattern + ')', log_text, re.IGNORECASE)
    if km:
        api_key = km.group(1)
    elif not api_key:
        km2 = APIKEY_RE.search(log_text)
        if km2:
            api_key = km2.group(0)
    mm = MODEL_RE.search(log_text)
    if mm:
        model = mm.group(1)
    return {'endpoint': endpoint, 'api_key': api_key, 'model': model}
