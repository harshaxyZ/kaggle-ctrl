from __future__ import annotations
import os
import threading
import time
import accounts
import server_manager as sm
_LOCK = threading.RLock()
_thread: threading.Thread | None = None
_running = False
_DESIRED: dict[str, str | None] = {'model_key': None}
_recovery_requested = False
TICK_S = 60
KAGGLE_LIMIT_S = 12 * 3600
PREEMPT_MARGIN_S = 45 * 60
BOOT_GRACE_S = 40 * 60
DOWN_GRACE_S = 5 * 60
POOL_TARGET = int(os.environ.get('KGCTRL_POOL_TARGET', '1'))
BOOT_RESERVE_H = 0.75

def state() -> dict:
    with _LOCK:
        return {'enabled': _running}

def start(model_key: str | None=None) -> dict:
    global _thread, _running
    with _LOCK:
        if model_key:
            _DESIRED['model_key'] = model_key
        if _running:
            return {'ok': True, 'message': 'already on', **state()}
        _running = True
    _thread = threading.Thread(target=_loop, daemon=True)
    _thread.start()
    return {'ok': True, 'message': 'keepalive on', **state()}

def stop() -> dict:
    global _running
    with _LOCK:
        _running = False
    return {'ok': True, 'message': 'keepalive off', **state()}

def _loop() -> None:
    while True:
        with _LOCK:
            if not _running:
                return
        try:
            step()
        except Exception:
            pass
        time.sleep(TICK_S)

def request_recovery() -> dict:
    """Ask the supervisor to bring back a missing or sick deployment.

    The router calls this when it cannot serve a request, which is the one
    moment where starting a kernel is worth the GPU time: there is no
    traffic being served anyway. The supervisor loop does the real work on
    its next tick, so this only nudges it.
    """
    global _recovery_requested
    with _LOCK:
        _recovery_requested = True
    return {'ok': True, 'requested': True}

def recovery_requested() -> bool:
    with _LOCK:
        return bool(_recovery_requested)

def clear_recovery_request() -> None:
    global _recovery_requested
    with _LOCK:
        _recovery_requested = False

def step() -> dict:
    """One supervision decision, per account. Exposed for tests.

    The goal is one healthy kernel on every usable account, which is what
    round robin needs. Supervision is therefore per account: a second kernel
    on a *different* account is wanted, while a second kernel on the *same*
    account is the brief overlap of a handoff and should be retired.
    """
    model_key = _DESIRED['model_key'] or sm.DEFAULT_MODEL
    try:
        sm.reconcile_with_kaggle()
    except Exception:
        pass
    srv = sm.state()
    deps = [d for d in srv['deployments'] if d['status'] != 'STOPPED']
    live = [d for d in deps if d['status'] == 'RUNNING' and d['endpoint']]
    booting = [d for d in deps if d['status'] in ('STARTING', 'QUEUED') or (d['status'] == 'RUNNING' and (not d['endpoint']))]
    broken = [d for d in deps if d['status'] == 'ERROR']
    actions = []

    def best_account(exclude_id=None):
        """Next account to host the pool kernel: most remaining hours wins.

        Rotation across accounts is the point: when the current host is
        nearing the 12h cap or just died, the replacement goes to a
        DIFFERENT account with quota, so accounts cycle and rest.
        """
        cands = []
        for acct in accounts.load():
            if not acct.enabled or not acct.token:
                continue
            if acct.cooldown_until and acct.cooldown_until > time.time():
                continue
            if acct.remaining_hours <= BOOT_RESERVE_H:
                continue
            cands.append(acct)
        cands.sort(key=lambda a: a.remaining_hours, reverse=True)
        for a in cands:
            if exclude_id is not None and a.id == exclude_id:
                continue
            return a
        for a in cands:
            return a
        return None
    for d in broken:
        sm.stop(d['id'])
        actions.append({'action': 'cleared_error', 'deployment': d['id']})
    if len(live) > POOL_TARGET:
        live.sort(key=lambda d: d['started_at'], reverse=True)
        for d in live[POOL_TARGET:]:
            sm.stop(d['id'])
            actions.append({'action': 'retire_extra', 'stopped': d['id']})
        live = live[:POOL_TARGET]
    if booting:
        actions.append({'action': 'wait', 'reason': 'booting', 'deployments': [d['id'] for d in booting]})
        return {'actions': actions}
    for d in live:
        if d['uptime_s'] >= KAGGLE_LIMIT_S - PREEMPT_MARGIN_S:
            acct = best_account(exclude_id=d['account_id'])
            if acct is None:
                actions.append({'action': 'drained', 'reason': 'no account with quota for handoff'})
                continue
            res = sm.start(model_key=d['model_key'], account_id=acct.id)
            actions.append({'action': 'preempt_rotate', 'from_account': d['account_name'], 'to_account': acct.name, 'uptime_s': d['uptime_s'], 'result': res.get('message')})
            return {'actions': actions}
    if len(live) < POOL_TARGET:
        if deps and (not recovery_requested()):
            last = max((d['last_update'] for d in deps))
            if time.time() - last < DOWN_GRACE_S and (not live):
                actions.append({'action': 'wait', 'reason': 'recently down'})
                return {'actions': actions}
        acct = best_account()
        if acct is None:
            actions.append({'action': 'drained', 'reason': 'every account out of quota until reset'})
            return {'actions': actions}
        res = sm.start(model_key=model_key, account_id=acct.id)
        actions.append({'action': 'start', 'account': acct.name, 'result': res.get('message')})
        return {'actions': actions}
    actions.append({'action': 'healthy', 'live': [d['id'] for d in live]})
    if any((a.get('action') == 'start' for a in actions)):
        clear_recovery_request()
    return {'actions': actions} if actions else {'action': 'idle'}
