"""
analytics.py - token and request accounting for the router.

Every routed request appends one row: model, account, prompt tokens, completion
tokens, latency, status. The dashboard reads aggregates from here. Rows are
capped so the file cannot grow without bound.
"""
from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
USAGE_FILE = HERE / 'usage.json'
MAX_ROWS = 5000
_LOCK = threading.RLock()

def _load() -> list[dict]:
    if not USAGE_FILE.exists():
        return []
    try:
        data = json.loads(USAGE_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _save(rows: list[dict]) -> None:
    tmp = USAGE_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rows[-MAX_ROWS:]), encoding='utf-8')
    os.replace(tmp, USAGE_FILE)

def record(model: str, account: str, deployment_id: str, prompt_tokens: int, completion_tokens: int, latency_ms: float, ok: bool=True) -> None:
    row = {'ts': time.time(), 'model': model, 'account': account, 'deployment': deployment_id, 'in': int(prompt_tokens or 0), 'out': int(completion_tokens or 0), 'ms': round(float(latency_ms or 0), 1), 'ok': bool(ok)}
    with _LOCK:
        rows = _load()
        rows.append(row)
        _save(rows)

def summary(window_s: float | None=None) -> dict:
    """Totals plus per-model and per-account breakdowns."""
    with _LOCK:
        rows = _load()
    if window_s:
        cutoff = time.time() - window_s
        rows = [r for r in rows if r.get('ts', 0) >= cutoff]
    total_in = sum((r.get('in', 0) for r in rows))
    total_out = sum((r.get('out', 0) for r in rows))
    ok = sum((1 for r in rows if r.get('ok')))
    failed = len(rows) - ok
    lat = [r['ms'] for r in rows if r.get('ms')]
    avg_ms = round(sum(lat) / len(lat), 1) if lat else 0.0
    MIN_GEN_MS = 200
    tps_samples = [r['out'] / (r['ms'] / 1000.0) for r in rows if r.get('ms', 0) >= MIN_GEN_MS and r.get('out') and r.get('ok')]
    avg_tps = round(sum(tps_samples) / len(tps_samples), 1) if tps_samples else 0.0
    peak_tps = round(max(tps_samples), 1) if tps_samples else 0.0
    if len(rows) >= 2:
        span = max(60.0, rows[-1].get('ts', 0) - rows[0].get('ts', 0))
        tpm = round(total_out / (span / 60.0), 1)
        rpm = round(len(rows) / (span / 60.0), 2)
    else:
        tpm = rpm = 0.0
    by_model: dict[str, dict] = {}
    by_account: dict[str, dict] = {}
    for r in rows:
        for bucket, key in ((by_model, r.get('model', '?')), (by_account, r.get('account', '?'))):
            b = bucket.setdefault(key, {'requests': 0, 'in': 0, 'out': 0, '_tps': [], '_ms': []})
            b['requests'] += 1
            b['in'] += r.get('in', 0)
            b['out'] += r.get('out', 0)
            if r.get('ms'):
                b['_ms'].append(r['ms'])
                if r.get('out') and r.get('ok'):
                    b['_tps'].append(r['out'] / (r['ms'] / 1000.0))
    for bucket in (by_model, by_account):
        for b in bucket.values():
            tps = b.pop('_tps', [])
            ms = b.pop('_ms', [])
            b['avg_tps'] = round(sum(tps) / len(tps), 1) if tps else 0.0
            b['avg_ms'] = round(sum(ms) / len(ms), 1) if ms else 0.0
    now = time.time()
    hourly = [0] * 24
    for r in rows:
        age_h = int((now - r.get('ts', now)) // 3600)
        if 0 <= age_h < 24:
            hourly[23 - age_h] += 1
    return {'requests': len(rows), 'ok': ok, 'failed': failed, 'tokens_in': total_in, 'tokens_out': total_out, 'tokens_total': total_in + total_out, 'avg_latency_ms': avg_ms, 'avg_tps': avg_tps, 'peak_tps': peak_tps, 'tpm': tpm, 'rpm': rpm, 'by_model': by_model, 'by_account': by_account, 'hourly': hourly, 'recent': rows[-25:][::-1]}

def clear() -> None:
    with _LOCK:
        _save([])
