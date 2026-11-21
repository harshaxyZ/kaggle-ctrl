from __future__ import annotations
import json
import hashlib
import hmac
import secrets
import os
import re
import shutil
import threading
import time
import requests
import uuid
from dataclasses import dataclass, field
from pathlib import Path
import accounts
import kaggle_client as kc
HERE = Path(__file__).resolve().parent
KERNEL_ROOT = HERE / 'serving_kernel'
STATE_FILE = HERE / 'server_state.json'
_LOCK = threading.RLock()
SUPERVISORS: dict[str, threading.Thread] = {}

@dataclass
class ModelSpec:
    key: str
    label: str
    hf_repo: str
    filename: str
    kind: str = 'gguf'
    ctx: int = 16384
    gpu_layers: int = -1
    mmproj: str = ''
    mtp: str = ''
    n_parallel: int = 2
    test_minutes: int = 0
    est_gb: float = 0.0
    tps: float | None = None
    notes: str = ''

    def as_dict(self) -> dict:
        return {'key': self.key, 'label': self.label, 'hf_repo': self.hf_repo, 'filename': self.filename, 'kind': self.kind, 'ctx': self.ctx, 'n_parallel': self.n_parallel, 'est_gb': self.est_gb, 'tps': self.tps, 'notes': self.notes}
from settings import SETTINGS
MODEL_FIELDS = set(ModelSpec.__dataclass_fields__)
MODELS = {row['key']: ModelSpec(**{k: v for k, v in row.items() if k in MODEL_FIELDS}) for row in SETTINGS['models']}
DEFAULT_MODEL = SETTINGS['default_model']
from kernel_template import generate_kernel

def _serve_script(spec, topic):
    custom = SETTINGS.get('kernel_templates', {}).get(spec.key)
    if custom:
        code = (HERE / custom).read_text()
        return code.replace('__INSTALL_TOPIC__', topic).replace('__INSTALL_KEY__', _backend_service_key())
    return generate_kernel(spec, topic, _backend_service_key())

@dataclass
class Deployment:
    id: str
    account_id: str
    account_name: str
    model_key: str
    kernel_ref: str = ''
    status: str = 'STARTING'
    endpoint: str = ''
    api_key: str = ''
    model_id: str = ''
    accelerator: str = 'NvidiaTeslaT4'
    started_at: float = 0.0
    last_update: float = 0.0
    message: str = ''
    logs: str = ''
    phase: str = ''

    def as_dict(self) -> dict:
        spec = MODELS.get(self.model_key, MODELS[DEFAULT_MODEL])
        return {'id': self.id, 'account_id': self.account_id, 'account_name': self.account_name, 'model_key': self.model_key, 'model_spec': spec.as_dict(), 'kernel_ref': self.kernel_ref, 'status': self.status, 'endpoint': self.endpoint, 'api_key': self.api_key, 'model_id': self.model_id or spec.filename, 'accelerator': self.accelerator, 'started_at': self.started_at, 'uptime_s': int(time.time() - self.started_at) if self.started_at else 0, 'last_update': self.last_update, 'message': self.message, 'logs_tail': self.logs[-6000:] if self.logs else '', 'phase': self.phase}
DEPLOYMENTS: dict[str, Deployment] = {}

def _save_state() -> None:
    """Persist the deployment registry.

    Merge-on-write, not overwrite. Two dashboard processes (a leftover from a
    previous run plus a freshly started one) each keep their own in-memory
    registry, and whichever saves last wins. A blind overwrite therefore
    deletes the other process's entries - including the record of a kernel
    that is still serving - which is how the dashboard ended up reporting zero
    live endpoints while a model was up.

    The newest source-validated state wins. A newer explicit stop must not
    be replaced by an old row merely because the old row has an endpoint.
    """
    with _LOCK:
        mine = {d.id: d.as_dict() for d in DEPLOYMENTS.values()}
    merged = dict(mine)
    try:
        if STATE_FILE.exists():
            rows = json.loads(STATE_FILE.read_text(encoding='utf-8'))
            if isinstance(rows, list):
                for r in rows:
                    rid = r.get('id')
                    if not rid:
                        continue
                    if rid not in merged:
                        merged[rid] = r
                    elif _prefer(r, mine[rid]):
                        merged[rid] = r
    except Exception:
        pass
    try:
        tmp = STATE_FILE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(list(merged.values()), indent=2), encoding='utf-8')
        os.replace(tmp, STATE_FILE)
    except Exception:
        pass

def _prefer(candidate: dict, current: dict) -> bool:
    """Choose newer evidence, never resurrect old RUNNING rows after a stop."""
    return float(candidate.get('last_update') or 0) > float(current.get('last_update') or 0)

def load_state() -> None:
    if not STATE_FILE.exists():
        return
    try:
        rows = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return
    if not isinstance(rows, list):
        return
    with _LOCK:
        for r in rows:
            if not r.get('id'):
                continue
            d = Deployment(id=r['id'], account_id=r.get('account_id', ''), account_name=r.get('account_name', ''), model_key=r.get('model_key', DEFAULT_MODEL), kernel_ref=r.get('kernel_ref', ''), status=r.get('status', 'STOPPED'), endpoint=r.get('endpoint', ''), api_key=r.get('api_key', ''), model_id=r.get('model_id', ''), accelerator=r.get('accelerator', 'NvidiaTeslaT4'), started_at=r.get('started_at', 0.0), last_update=r.get('last_update', 0.0), message=r.get('message', ''), logs=r.get('logs_tail', ''))
            DEPLOYMENTS[d.id] = d

def _workdir(deployment_id: str) -> Path:
    d = KERNEL_ROOT / deployment_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def render_kernel(spec: ModelSpec, account: accounts.Account, deployment_id: str) -> Path:
    workdir = _workdir(deployment_id)
    slug = f'serve-{spec.key}-{deployment_id[-8:]}'
    kernel_id = f"{account.username or 'me'}/{slug}"
    meta = {'id': kernel_id, 'title': slug, 'code_file': 'serve.py', 'language': 'python', 'kernel_type': 'script', 'is_private': 'true', 'enable_gpu': 'true', 'enable_tpu': 'false', 'enable_internet': 'true', 'machine_shape': 'NvidiaTeslaT4'}
    ready = Path(__file__).resolve().parent / 'weights_persistence_ready.json'
    if ready.exists():
        cache = json.loads(ready.read_text()).get('gpu', {}).get(account.id)
        if cache and cache.get('verified'):
            meta['dataset_sources'] = [cache['ref']]
    (workdir / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    topic = _topic_for(deployment_id)
    installer_src = ''
    inst = Path(__file__).resolve().parent / 'llama_installer.py'
    if inst.exists():
        installer_src = inst.read_text(encoding='utf-8')
    script = installer_src + '\n\n' + _serve_script(spec, topic)
    auth_path = HERE / 'deployment_event_auth.json'
    with _LOCK:
        auth = json.loads(auth_path.read_text()) if auth_path.exists() else {}
        signing_key = auth.setdefault(deployment_id, secrets.token_hex(32))
        auth_path.write_text(json.dumps(auth))
        auth_path.chmod(384)
    signature_setup = 'import hmac, hashlib\nEVENT_SIGNING_KEY = ' + repr(signing_key) + '\n'
    backend_key = re.search('^API_KEY = "([^"]+)"', script, re.M)
    with _LOCK:
        if backend_key and deployment_id in DEPLOYMENTS:
            DEPLOYMENTS[deployment_id].api_key = backend_key.group(1)
    script = script.replace('HF_REPO = ', signature_setup + 'HF_REPO = ', 1)
    script = script.replace('    body = json.dumps({"stage": stage, "msg": msg, "ts": time.time(), **extra}).encode()', '    event = {"stage": stage, "msg": msg, "ts": time.time(), **extra}\n    event["signature"] = hmac.new(EVENT_SIGNING_KEY.encode(), json.dumps(event, sort_keys=True, separators=(chr(44),chr(58))).encode(), hashlib.sha256).hexdigest()\n    body = json.dumps(event).encode()')
    (workdir / 'serve.py').write_text(script, encoding='utf-8')
    return workdir

def _topic_for(deployment_id: str) -> str:
    """Stable per-deployment status topic. Anyone who knows the id can read it,
    which is fine: it carries only a tunnel URL that is already public."""
    return 'kgctrl-' + deployment_id.replace('dep-', '')

def topic_for_slug(slug: str) -> str:
    """Inverse of _topic_for: recover the ntfy topic from a kernel slug.

    Kernel slugs are built as f"serve-{model_key}-{deployment_id[-8:]}" and the
    topic is "kgctrl-" + the same 8 hex chars. So a running kernel's topic can
    always be derived from its slug, which is what lets the dashboard adopt a
    kernel it has no local record of.
    """
    m = re.search('serve-[^-]+-([0-9a-f]{8})$', slug)
    if not m:
        return ''
    return 'kgctrl-' + m.group(1)

def _slug_to_model_key(slug: str) -> str:
    """Recover the model key from a kernel slug: serve-<model_key>-<suffix>."""
    m = re.match('serve-(.+)-[0-9a-f]{8}$', slug)
    return m.group(1) if m else DEFAULT_MODEL

def account_serving_kernels(acct: accounts.Account) -> list[tuple[str, str]]:
    """Serving kernels that Kaggle says exist on this account.

    Returns (kernel_ref, kernel_status) pairs for kernels that match our
    serve-* naming, so a caller can tell "this account already has a model
    booting" without trusting local state.
    """
    out: list[tuple[str, str]] = []
    try:
        refs = kc.list_kernels(account_username=acct.username, config_dir=str(acct.config_dir()), token=acct.token)
    except Exception:
        return out
    for ref in refs:
        slug = ref.split('/')[-1]
        if not slug.startswith('serve-'):
            continue
        try:
            st = kc.kernel_status(ref, config_dir=str(acct.config_dir()), token=acct.token)
        except Exception:
            st = 'ERROR'
        out.append((ref, st))
    return out

def adopt_running_kernels() -> list[dict]:
    """Find serving kernels that Kaggle says are alive but this dashboard does
    not know about, and adopt them.

    Why this exists: the dashboard's deployment registry lives in
    server_state.json, and the dashboard can be restarted (or a second copy can
    run) while a kernel is still serving. The kernel keeps running and keeps
    its tunnel, but the new process has no record of it, so the router sees
    zero live endpoints and answers 503 while a perfectly good model is live.

    Rather than trust local state, we ask Kaggle what is actually running and
    re-derive the ntfy topic from the kernel slug. The topic carries the
    endpoint and the model API key, so an adopted kernel is immediately
    routable.
    """
    known = {d.kernel_ref for d in DEPLOYMENTS.values() if d.kernel_ref}
    adopted: list[dict] = []
    for acct in accounts.load():
        if not acct.token:
            continue
        try:
            refs = kc.list_kernels(account_username=acct.username, config_dir=str(acct.config_dir()), token=acct.token)
        except Exception:
            continue
        for ref in refs:
            if ref in known:
                continue
            slug = ref.split('/')[-1]
            topic = topic_for_slug(slug)
            if not topic:
                continue
            status = kc.kernel_status(ref, config_dir=str(acct.config_dir()), token=acct.token)
            if 'RUNNING' not in status and 'QUEUED' not in status:
                continue
            dep_id = 'dep-' + topic.replace('kgctrl-', '')
            model_key = _slug_to_model_key(slug)
            with _LOCK:
                if dep_id in DEPLOYMENTS:
                    continue
                DEPLOYMENTS[dep_id] = Deployment(id=dep_id, account_id=acct.id, account_name=acct.name, model_key=model_key, kernel_ref=ref, status='RUNNING' if 'RUNNING' in status else 'QUEUED', accelerator='NvidiaTeslaT4', last_update=time.time(), message='adopted from Kaggle after dashboard restart')
            poll_status(dep_id)
            adopted.append({'id': dep_id, 'kernel_ref': ref, 'account': acct.name})
    if adopted:
        _save_state()
    return adopted

def reconcile_with_kaggle() -> dict:
    """Align the local registry with what Kaggle actually has running.

    Three things are reconciled:
      * a kernel Kaggle still runs but we have no record of -> adopted
      * a kernel we think is running but Kaggle has stopped -> marked stopped
      * a kernel that is RUNNING but has not published an endpoint yet -> kept
        as STARTING, because it is booting and the account is occupied

    That last one matters: Kaggle allows only a couple of batch kernels per
    account, and pushing a new one can cancel the one that is still booting.
    A booting kernel must therefore always count as "account busy".
    """
    result = {'adopted': [], 'stopped': [], 'booting': []}
    for acct in accounts.load():
        if not acct.token:
            continue
        try:
            refs = kc.list_kernels(account_username=acct.username, config_dir=str(acct.config_dir()), token=acct.token)
        except Exception:
            continue
        with _LOCK:
            for dep in list(DEPLOYMENTS.values()):
                if dep.account_id != acct.id:
                    continue
                if dep.status not in ('RUNNING', 'STARTING', 'QUEUED'):
                    continue
                if not dep.kernel_ref:
                    continue
                if dep.kernel_ref not in refs:
                    dep.status = 'STOPPED'
                    dep.endpoint = ''
                    dep.api_key = ''
                    dep.message = 'kernel gone from Kaggle (deleted or capped)'
                    dep.last_update = time.time()
                    result['stopped'].append(dep.id)
        for ref in refs:
            slug = ref.split('/')[-1]
            topic = topic_for_slug(slug)
            if not topic:
                continue
            dep_id = 'dep-' + topic.replace('kgctrl-', '')
            try:
                st = kc.kernel_status(ref, config_dir=str(acct.config_dir()), token=acct.token)
            except Exception:
                st = 'ERROR'
            alive = 'RUNNING' in st or 'QUEUED' in st
            with _LOCK:
                dep = DEPLOYMENTS.get(dep_id)
                if dep is None:
                    continue
                if not alive:
                    if dep.status in ('RUNNING', 'STARTING', 'QUEUED'):
                        dep.status = 'STOPPED'
                        dep.endpoint = ''
                        dep.api_key = ''
                        dep.message = f"kernel {st.replace('KernelWorkerStatus.', '')}"
                        result['stopped'].append(dep_id)
                elif dep.status == 'RUNNING' and (not dep.endpoint):
                    dep.status = 'STARTING'
                    dep.phase = 'booting'
                    result['booting'].append(dep_id)
    try:
        result['adopted'] = adopt_running_kernels()
    except Exception:
        pass
    _save_state()
    return result

def _backend_service_key():
    path = Path(__file__).resolve().parent / 'backend_service_key.json'
    if not path.exists():
        path.write_text(json.dumps({'key': 'sk-backend-' + secrets.token_urlsafe(32)}))
        path.chmod(384)
    value = json.loads(path.read_text())['key']
    if not value or any((c in value for c in ('"', '\n', '\r'))):
        raise ValueError('Invalid backend key')
    return value

def start(model_key: str | None=None, account_id: str | None=None, accelerator: str='NvidiaTeslaT4') -> dict:
    """Push one serving kernel on one account. Non-blocking."""
    model_key = model_key or DEFAULT_MODEL
    spec = MODELS.get(model_key)
    if not spec:
        return {'ok': False, 'message': f'unknown model_key {model_key}'}
    if not spec.hf_repo or not spec.filename:
        return {'ok': False, 'message': 'Configure a model repository and filename in local settings first'}
    with _LOCK:
        active = {d.account_id for d in DEPLOYMENTS.values() if d.status in ('STARTING', 'QUEUED', 'RUNNING')}
    if account_id:
        acct = accounts.get(account_id)
        if not acct:
            return {'ok': False, 'message': f'unknown account {account_id}'}
        if not acct.usable:
            return {'ok': False, 'message': f'account {acct.name} not usable (disabled, cooling down, or out of hours)'}
        if account_id in active:
            return {'ok': False, 'message': f'account {acct.name} already hosting a kernel'}
        existing = [(ref, st) for ref, st in account_serving_kernels(acct) if 'RUNNING' in st or 'QUEUED' in st]
        if existing:
            reconcile_with_kaggle()
            return {'ok': False, 'message': f'account {acct.name} already has a live kernel ({existing[0][0]}); not pushing a second one'}
    else:
        counts: dict[str, int] = {}
        with _LOCK:
            for d in DEPLOYMENTS.values():
                if d.status in ('STARTING', 'QUEUED', 'RUNNING'):
                    counts[d.account_id] = counts.get(d.account_id, 0) + 1
        acct = accounts.pick_for_deploy(counts)
        if not acct:
            return {'ok': False, 'message': 'no usable account. add one or wait for cooldown/hours reset'}
    dep_id = 'dep-' + uuid.uuid4().hex[:8]
    dep = Deployment(id=dep_id, account_id=acct.id, account_name=acct.name, model_key=model_key, accelerator=accelerator, status='STARTING', message=f'pushing {spec.label} on account {acct.name}', started_at=time.time(), last_update=time.time())
    with _LOCK:
        DEPLOYMENTS[dep_id] = dep
    _save_state()
    config_dir = str(acct.config_dir())
    workdir = render_kernel(spec, acct, dep_id)
    out = kc.kernel_push(str(workdir), accelerator=accelerator, timeout=300, config_dir=config_dir, token=acct.token)
    if 'kernel push error' in out.lower() or out.startswith('push failed'):
        with _LOCK:
            dep.status = 'ERROR'
            dep.message = out.strip()[:300]
            dep.logs = f'push: {out}'
            dep.last_update = time.time()
        _save_state()
        accounts.record_failure(acct.id)
        return {'ok': False, 'deployment_id': dep_id, 'account': acct.name, 'message': out.strip()[:300]}
    m = re.search('https://www\\.kaggle\\.com/code/(\\S+)', out)
    default_slug = f"{acct.username or 'me'}/serve-{spec.key}-{dep_id[-8:]}"
    with _LOCK:
        if m:
            dep.kernel_ref = m.group(1)
        else:
            dep.kernel_ref = default_slug
        dep.logs = f'push: {out}\n(ref={dep.kernel_ref})'
        dep.last_update = time.time()
    _save_state()
    t = threading.Thread(target=_supervise, args=(dep_id,), daemon=True)
    SUPERVISORS[dep_id] = t
    t.start()
    return {'ok': True, 'deployment_id': dep_id, 'account': acct.name, 'message': f'kernel pushed on {acct.name}; queued for GPU'}

def _supervise(dep_id: str) -> None:
    deadline = time.time() + 60 * 60
    config_dir = None
    while time.time() < deadline:
        time.sleep(20)
        with _LOCK:
            dep = DEPLOYMENTS.get(dep_id)
            if not dep:
                return
            ref = dep.kernel_ref
            acct_id = dep.account_id
            status = dep.status
        if status == 'STOPPED':
            return
        acct = accounts.get(acct_id)
        if acct:
            config_dir = str(acct.config_dir())
        st = kc.kernel_status(ref, config_dir=config_dir, token=acct.token if acct else None)
        with _LOCK:
            if dep.status == 'STOPPED':
                return
            if 'QUEUED' in st:
                dep.status = 'QUEUED'
            elif 'RUNNING' in st:
                dep.status = 'RUNNING'
            elif any((x in st for x in ('COMPLETE', 'ERROR', 'FAIL'))):
                dep.status = 'ERROR'
            dep.message = f"kernel {st.replace('KernelWorkerStatus.', '')}"
            dep.last_update = time.time()
        _save_state()
        poll_status(dep_id)
        with _LOCK:
            dep = DEPLOYMENTS.get(dep_id)
            done = dep and dep.endpoint
            failed = dep and dep.status == 'ERROR'
        if done:
            accounts.record_success(acct_id)
            return
        if failed or any((x in st for x in ('ERROR', 'FAIL', 'COMPLETE'))):
            accounts.record_failure(acct_id)
            return

def poll_status(dep_id: str) -> dict:
    """Read the deployment's ntfy topic for status.

    The topic is append-only, so the newest event is often a heartbeat that
    carries no endpoint. Scanning only the last event therefore loses the
    endpoint, and the deployment stays RUNNING but unroutable. Instead we
    walk the whole topic and take the newest event that actually carries the
    information we need.
    """
    topic = _topic_for(dep_id)
    try:
        r = requests.get(f'https://ntfy.sh/{topic}/json', params={'poll': '1', 'since': 'all'}, timeout=15)
        if not r.ok:
            return {}
    except Exception:
        return {}
    events = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get('event') == 'message' and ev.get('message'):
            try:
                event = json.loads(ev['message'])
                auth_path = HERE / 'deployment_event_auth.json'
                auth = json.loads(auth_path.read_text()) if auth_path.exists() else {}
                signing_key = auth.get(dep_id)
                if signing_key:
                    signature = event.pop('signature', '')
                    expected = hmac.new(signing_key.encode(), json.dumps(event, sort_keys=True, separators=(',', ':')).encode(), hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(signature, expected):
                        continue
                else:
                    event.pop('endpoint', None)
                    event.pop('api_key', None)
                events.append(event)
            except Exception:
                continue
    if not events:
        return {}
    events.sort(key=lambda e: e.get('ts', 0))
    latest = events[-1]
    live = None
    for e in events:
        if e.get('endpoint'):
            live = e
    err = None
    for e in events:
        if e.get('stage') == 'error':
            err = e
    with _LOCK:
        dep = DEPLOYMENTS.get(dep_id)
        if not dep or dep.status not in ('STARTING', 'QUEUED', 'RUNNING'):
            return latest
        stage = latest.get('stage', '')
        dep.phase = stage
        dep.message = latest.get('msg', stage)
        if live:
            dep.endpoint = live['endpoint']
            dep.api_key = dep.api_key
            dep.model_id = live.get('model', dep.model_id)
            if live.get('started'):
                dep.started_at = float(live['started'])
            dep.status = 'RUNNING'
            dep.message = live.get('msg', 'serving')
        elif err:
            dep.status = 'ERROR'
            dep.message = err.get('detail') or err.get('msg', 'error')
        elif stage in ('boot', 'install', 'install_ok', 'download', 'download_ok', 'loading', 'ready'):
            if dep.status not in ('RUNNING',):
                dep.status = 'STARTING'
        dep.last_update = time.time()
    _save_state()
    return latest

def _scrape(dep_id: str, config_dir: str | None=None) -> None:
    poll_status(dep_id)

def stop(dep_id: str | None=None) -> dict:
    """Stop one deployment, or all when dep_id is None."""
    with _LOCK:
        targets = [DEPLOYMENTS[dep_id]] if dep_id in DEPLOYMENTS else list(DEPLOYMENTS.values()) if dep_id is None else []
    if not targets:
        return {'ok': False, 'message': 'no such deployment'}
    results = []
    for dep in targets:
        acct = accounts.get(dep.account_id)
        config_dir = str(acct.config_dir()) if acct else None
        ref = dep.kernel_ref
        res = kc.kernel_delete(ref, config_dir=config_dir, token=acct.token if acct else None) if ref else 'no kernel ref'
        with _LOCK:
            dep.status = 'STOPPED'
            dep.endpoint = ''
            dep.api_key = ''
            dep.started_at = 0.0
            dep.message = res
            dep.last_update = time.time()
        results.append({'deployment_id': dep.id, 'account': dep.account_name, 'kernel_ref': ref, 'result': res})
    _save_state()
    return {'ok': True, 'stopped': results}

def refresh() -> dict:
    """Poll status for every active deployment."""
    try:
        reconcile_with_kaggle()
    except Exception:
        pass
    with _LOCK:
        active = [(d.id, d.kernel_ref, d.account_id) for d in DEPLOYMENTS.values() if d.status in ('STARTING', 'QUEUED', 'RUNNING')]
    for dep_id, ref, acct_id in active:
        if not ref:
            continue
        acct = accounts.get(acct_id)
        config_dir = str(acct.config_dir()) if acct else None
        st = kc.kernel_status(ref, config_dir=config_dir, token=acct.token if acct else None)
        with _LOCK:
            dep = DEPLOYMENTS.get(dep_id)
            if not dep:
                continue
            if 'QUEUED' in st:
                dep.status = 'QUEUED'
            elif 'RUNNING' in st:
                dep.status = 'RUNNING'
            elif any((x in st for x in ('COMPLETE', 'ERROR', 'FAIL'))):
                dep.status = 'ERROR'
            dep.message = f"kernel {st.replace('KernelWorkerStatus.', '')}"
            dep.last_update = time.time()
        poll_status(dep_id)
    for acct in accounts.load():
        try:
            q = kc.get_accelerator_quota(token=acct.token, config_dir=str(acct.config_dir()))
            if not q.error:
                accounts.refresh_quota(acct.id, q.gpu_used_h, q.gpu_total_h, q.gpu_refresh_at)
        except Exception:
            pass
    return state()

def state() -> dict:
    with _LOCK:
        deps = [d.as_dict() for d in DEPLOYMENTS.values()]
        counts: dict[str, int] = {}
        for d in DEPLOYMENTS.values():
            if d.status in ('STARTING', 'QUEUED', 'RUNNING'):
                counts[d.account_id] = counts.get(d.account_id, 0) + 1
    live = [d for d in deps if d['status'] == 'RUNNING' and d['endpoint']]
    pool = accounts.pool_summary(counts)
    pool['enabled'] = pool.get('enabled', 0)
    primary = None
    if live:
        primary = sorted(live, key=lambda d: -d.get('uptime_s', 0))[0]
    return {'deployments': deps, 'live_endpoints': live, 'pool': pool, 'active_by_account': counts, 'models': [m.as_dict() for m in MODELS.values()], 'status': primary['status'] if primary else 'STOPPED', 'endpoint': primary['endpoint'] if primary else '', 'api_key': primary['api_key'] if primary else '', 'model_id': primary['model_id'] if primary else '', 'model_key': primary['model_key'] if primary else '', 'account_name': primary['account_name'] if primary else '', 'uptime_s': primary['uptime_s'] if primary else 0, 'started_at': primary['started_at'] if primary else 0, 'message': primary['message'] if primary else '', 'phase': primary['phase'] if primary else '', 'logs': primary['logs_tail'] if primary else ''}
load_state()
