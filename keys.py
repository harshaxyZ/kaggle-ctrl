"""
keys.py - dashboard API keys with names, expiry, and per-key usage.

A key gates both the control API and the /v1 router. Keys can be revoked
(kept for audit, stops working) or deleted outright. Each key tracks its own
request and token counts so the dashboard can show usage per key.
"""
from __future__ import annotations
import json
import os
import secrets
import threading
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
KEYS_FILE = HERE / 'api_keys.json'
_LOCK = threading.RLock()

def _load() -> list[dict]:
    if not KEYS_FILE.exists():
        return []
    try:
        data = json.loads(KEYS_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _save(rows: list[dict]) -> None:
    tmp = KEYS_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    os.replace(tmp, KEYS_FILE)

def _new_key() -> str:
    return 'sk-kctrl-' + secrets.token_urlsafe(32)

def create(name: str='', expires_days: float | None=None) -> dict:
    now = time.time()
    row = {'key': _new_key(), 'name': (name or '').strip() or f'key-{int(now)}', 'created_at': now, 'expires_at': now + expires_days * 86400 if expires_days else None, 'last_used': 0.0, 'revoked': False, 'requests': 0, 'tokens_in': 0, 'tokens_out': 0}
    with _LOCK:
        rows = _load()
        rows.append(row)
        _save(rows)
    return row

def _expired(row: dict) -> bool:
    exp = row.get('expires_at')
    return bool(exp) and time.time() > exp

def valid(row: dict) -> bool:
    return not row.get('revoked') and (not _expired(row))

def is_valid(key: str) -> bool:
    if not key:
        return False
    with _LOCK:
        rows = _load()
        for r in rows:
            if r['key'] == key:
                if not valid(r):
                    return False
                r['last_used'] = time.time()
                _save(rows)
                return True
    return False

def get(key: str) -> dict | None:
    for r in _load():
        if r['key'] == key:
            return r
    return None

def record_usage(key: str, tokens_in: int, tokens_out: int) -> None:
    """Attribute a request to a key.

    If the key is not in the registry (it may have been issued by another
    process, or the registry was reset) create the row rather than dropping
    the usage on the floor, so per-key accounting never silently loses data.
    """
    if not key:
        return
    with _LOCK:
        rows = _load()
        found = False
        for r in rows:
            if r['key'] == key:
                found = True
                r['requests'] = r.get('requests', 0) + 1
                r['tokens_in'] = r.get('tokens_in', 0) + int(tokens_in or 0)
                r['tokens_out'] = r.get('tokens_out', 0) + int(tokens_out or 0)
                r['last_used'] = time.time()
        if not found:
            rows.append({'key': key, 'name': 'unknown', 'created_at': time.time(), 'expires_at': None, 'last_used': time.time(), 'revoked': False, 'requests': 1, 'tokens_in': int(tokens_in or 0), 'tokens_out': int(tokens_out or 0)})
        _save(rows)
IN_USE_WINDOW_S = 300

def all_keys() -> list[dict]:
    rows = _load()
    now = time.time()
    out = []
    for r in rows:
        r = dict(r)
        if r.get('revoked'):
            r['state'] = 'revoked'
        elif _expired(r):
            r['state'] = 'expired'
        else:
            r['state'] = 'active'
        last = r.get('last_used') or 0
        age = now - last if last else None
        if age is not None and age <= IN_USE_WINDOW_S:
            r['in_use'] = 'active now'
        elif r['state'] != 'active':
            r['in_use'] = 'unusable'
        elif age is None:
            r['in_use'] = 'never used'
        elif age < 3600:
            r['in_use'] = f'used {int(age // 60)}m ago'
        elif age < 86400:
            r['in_use'] = f'used {int(age // 3600)}h ago'
        else:
            r['in_use'] = f'used {int(age // 86400)}d ago'
        r['expires_in_days'] = round((r['expires_at'] - now) / 86400, 1) if r.get('expires_at') else None
        out.append(r)
    return out

def delete_all() -> int:
    """Remove every key. Used to reset the dashboard to a clean state."""
    with _LOCK:
        rows = _load()
        n = len(rows)
        _save([])
    return n

def revoke(key: str) -> bool:
    with _LOCK:
        rows = _load()
        hit = False
        for r in rows:
            if r['key'] == key:
                r['revoked'] = True
                hit = True
        if hit:
            _save(rows)
        return hit

def unrevoke(key: str) -> bool:
    with _LOCK:
        rows = _load()
        hit = False
        for r in rows:
            if r['key'] == key:
                r['revoked'] = False
                hit = True
        if hit:
            _save(rows)
        return hit

def delete(key: str) -> bool:
    with _LOCK:
        rows = _load()
        kept = [r for r in rows if r['key'] != key]
        if len(kept) == len(rows):
            return False
        _save(kept)
        return True

def rename(key: str, name: str) -> bool:
    with _LOCK:
        rows = _load()
        for r in rows:
            if r['key'] == key:
                r['name'] = name.strip() or r['name']
                _save(rows)
                return True
    return False

def any_active() -> bool:
    return any((valid(r) for r in _load()))

def any_keys() -> bool:
    return any_active()

def generate(label: str='default') -> dict:
    """Kept for the old call sites; name instead of label."""
    row = create(label)
    return {'key': row['key'], 'label': row['name'], 'created_at': row['created_at']}
