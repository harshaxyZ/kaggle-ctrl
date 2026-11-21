"""
accounts.py - multi-account registry for Kaggle GPU pooling.

Each account contributes 30 GPU-hours/week. With N accounts you get N x 30 h.
The registry stores credentials, health, and usage; the scheduler picks which
account should host the next serving kernel and which live endpoint should
take the next request.

All state lives in accounts.json next to this file. Writes are atomic.
"""
from __future__ import annotations
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
ACCOUNTS_FILE = HERE / 'accounts.json'
KAGGLE_DIR = Path.home() / '.kaggle' / 'accounts'
_LOCK = threading.RLock()
WEEKLY_HOURS = 30.0

@dataclass
class Account:
    id: str
    name: str
    token: str
    username: str = ''
    enabled: bool = True
    weight: int = 1
    max_hours: float = WEEKLY_HOURS
    used_hours: float = 0.0
    reset_at: str = ''
    fail_count: int = 0
    cooldown_until: float = 0.0
    last_used: float = 0.0
    last_quota_check: float = 0.0
    notes: str = ''

    @property
    def remaining_hours(self) -> float:
        return max(0.0, self.max_hours - self.used_hours)

    @property
    def usable(self) -> bool:
        return self.enabled and bool(self.token) and (self.cooldown_until <= time.time()) and (self.remaining_hours > 0.05)

    def as_dict(self, reveal: bool=False) -> dict:
        d = asdict(self)
        if not reveal:
            d['token'] = mask_token(self.token)
        d['remaining_hours'] = round(self.remaining_hours, 2)
        d['usable'] = self.usable
        d['usage_pct'] = round(self.used_hours / self.max_hours * 100, 1) if self.max_hours else 0.0
        return d

    def config_dir(self) -> Path:
        """Per-account KAGGLE_CONFIG_DIR so the CLI authenticates as this user.

        The CLI reads kaggle.json for the username. A bare access_token file is
        not enough: without the username the CLI falls back to whatever the
        default config says and every push lands on the wrong account.
        """
        d = KAGGLE_DIR / self.id
        d.mkdir(parents=True, exist_ok=True)
        tok = d / 'access_token'
        if self.token.startswith('KGAT_'):
            if not tok.exists() or tok.read_text(encoding='utf-8').strip() != self.token:
                tok.write_text(self.token, encoding='utf-8')
            tok.chmod(384)
        elif tok.exists():
            tok.unlink()
        user = self.username or self._lookup_username()
        if user and (not self.username):
            self.username = user
            try:
                update(self.id, username=user)
            except Exception:
                pass
        if user:
            kj = d / 'kaggle.json'
            cfg = {'username': user, 'key': self.token}
            if not kj.exists() or json.loads(kj.read_text(encoding='utf-8')) != cfg:
                kj.write_text(json.dumps(cfg), encoding='utf-8')
            kj.chmod(384)
        return d

    def _lookup_username(self) -> str:
        """Ask Kaggle who this token belongs to."""
        try:
            import requests
            r = requests.get('https://www.kaggle.com/api/v1/hello', headers={'Authorization': f'Bearer {self.token}', 'User-Agent': 'kaggle-cli/2.2.5'}, timeout=20)
            if r.ok:
                return r.json().get('userName', '')
        except Exception:
            pass
        return ''

def mask_token(tok: str) -> str:
    if not tok:
        return ''
    if len(tok) <= 12:
        return tok[:4] + '...'
    return tok[:9] + '...' + tok[-4:]

def _load_raw() -> list[dict]:
    if not ACCOUNTS_FILE.exists():
        return []
    try:
        data = json.loads(ACCOUNTS_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_raw(rows: list[dict]) -> None:
    tmp = ACCOUNTS_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    os.replace(tmp, ACCOUNTS_FILE)

def load() -> list[Account]:
    with _LOCK:
        rows = _load_raw()
        if not rows:
            rows = [seed_primary()]
            _save_raw(rows)
    out: list[Account] = []
    for r in rows:
        known = {f for f in Account.__dataclass_fields__}
        out.append(Account(**{k: v for k, v in r.items() if k in known}))
    return out

def save(accounts: list[Account]) -> None:
    with _LOCK:
        _save_raw([asdict(a) for a in accounts])

def seed_primary() -> dict:
    tok = os.environ.get('KAGGLE_API_TOKEN', '').strip()
    if not tok:
        p = Path.home() / '.kaggle' / 'access_token'
        if p.exists():
            tok = p.read_text(encoding='utf-8').strip()
    return asdict(Account(id='acct-primary', name='primary', token=tok, username=os.environ.get('KAGGLE_USERNAME') or __import__('settings').SETTINGS.get('username', ''), notes='primary account'))

def add(name: str, token: str, username: str='', notes: str='') -> Account:
    accounts = load()
    acct = Account(id='acct-' + uuid.uuid4().hex[:8], name=name or f'acct-{len(accounts) + 1}', token=token.strip(), username=username.strip(), notes=notes)
    accounts.append(acct)
    save(accounts)
    return acct

def remove(account_id: str) -> bool:
    accounts = load()
    kept = [a for a in accounts if a.id != account_id]
    if len(kept) == len(accounts):
        return False
    save(kept)
    return True

def update(account_id: str, **fields) -> Account | None:
    accounts = load()
    for a in accounts:
        if a.id == account_id:
            for k, v in fields.items():
                if hasattr(a, k) and v is not None:
                    setattr(a, k, v)
            save(accounts)
            return a
    return None

def get(account_id: str) -> Account | None:
    for a in load():
        if a.id == account_id:
            return a
    return None

def get_by_token(token: str) -> Account | None:
    for a in load():
        if a.token == token:
            return a
    return None

def record_success(account_id: str) -> None:
    accounts = load()
    for a in accounts:
        if a.id == account_id:
            a.fail_count = 0
            a.cooldown_until = 0.0
            a.last_used = time.time()
    save(accounts)

def record_failure(account_id: str, cooldown_s: float=600) -> None:
    accounts = load()
    for a in accounts:
        if a.id == account_id:
            a.fail_count += 1
            backoff = min(cooldown_s * 2 ** (a.fail_count - 1), 6 * 3600)
            a.cooldown_until = time.time() + backoff
    save(accounts)

def record_hours(account_id: str, hours: float) -> None:
    if hours <= 0:
        return
    accounts = load()
    for a in accounts:
        if a.id == account_id:
            a.used_hours = min(a.max_hours, a.used_hours + hours)
            a.last_quota_check = time.time()
    save(accounts)

def refresh_quota(account_id: str, used_h: float, total_h: float, reset_at: str) -> None:
    accounts = load()
    for a in accounts:
        if a.id == account_id:
            a.used_hours = used_h
            a.max_hours = total_h or a.max_hours
            a.reset_at = reset_at or a.reset_at
            a.last_quota_check = time.time()
    save(accounts)

def pick_for_deploy(active_counts: dict[str, int] | None=None) -> Account | None:
    """
    Choose an account to host the next serving kernel.

    Ordering: fewest active kernels, then most remaining hours, then
    least-recently used. Skips accounts in cooldown or out of hours.
    """
    active_counts = active_counts or {}
    candidates = [a for a in load() if a.usable]
    if not candidates:
        return None
    candidates.sort(key=lambda a: (active_counts.get(a.id, 0), -a.remaining_hours * max(1, a.weight), a.last_used))
    return candidates[0]

def pool_summary(active_counts: dict[str, int] | None=None) -> dict:
    active_counts = active_counts or {}
    accounts = load()
    total_capacity = sum((a.max_hours for a in accounts if a.enabled))
    total_remaining = sum((a.remaining_hours for a in accounts if a.enabled))
    active = sum(active_counts.values())
    return {'accounts': len(accounts), 'enabled': sum((1 for a in accounts if a.enabled)), 'usable': sum((1 for a in accounts if a.usable)), 'total_capacity_h': round(total_capacity, 1), 'total_remaining_h': round(total_remaining, 1), 'active_kernels': active, 'slots_free': max(0, len(accounts) - active)}
