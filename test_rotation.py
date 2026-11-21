"""Unit tests for pool-rotation keepalive (no Kaggle calls - all mocked)."""
from __future__ import annotations
import time
from unittest import mock
import accounts
import keepalive
import server_manager as sm
PASS = []

def check(name, cond):
    PASS.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name)

def acct(aid, name, remaining, enabled=True, token='tok', cooldown=0.0):
    a = accounts.Account(id=aid, name=name, token=token, username=name, enabled=enabled, max_hours=30.0, used_hours=30.0 - remaining)
    a.cooldown_until = cooldown
    return a

def dep(did, acct_id, status='RUNNING', endpoint='https://x.trycloudflare.com', started=0.0, uptime=0.0):
    return {'id': did, 'account_id': acct_id, 'account_name': acct_id, 'model_key': 'configured-model', 'kernel_ref': f'u/serve-configured-model-{did[-8:]}', 'status': status, 'endpoint': endpoint, 'started_at': started, 'uptime_s': uptime, 'last_update': time.time(), 'phase': '', 'message': ''}

def run_step(accts, deps, recovery=False):
    m_start = mock.Mock(side_effect=lambda model_key=None, account_id=None: {'ok': True, 'message': f'started on {account_id}'})
    m_stop = mock.Mock(side_effect=lambda dep_id=None: {'ok': True})
    with mock.patch.object(sm, 'reconcile_with_kaggle', lambda: {}), mock.patch.object(sm, 'state', lambda: {'deployments': deps}), mock.patch.object(accounts, 'load', lambda: accts), mock.patch.object(sm, 'start', m_start), mock.patch.object(sm, 'stop', m_stop), mock.patch.object(keepalive, 'recovery_requested', lambda: recovery), mock.patch.object(keepalive, 'clear_recovery_request', lambda: None):
        out = keepalive.step()
    return (out, m_start, m_stop)
A = acct('acct-a', 'a', 20.0)
B = acct('acct-b', 'b', 25.0)
C = acct('acct-c', 'c', 5.0)
out, m_start, _ = run_step([A, B, C], [], recovery=True)
check('cold start picks fullest account', any((a.get('action') == 'start' and a.get('account') == 'b' for a in out['actions'])))
out, m_start, _ = run_step([A, B], [dep('dep-1', 'acct-a', uptime=100)], recovery=False)
check('young live kernel is healthy', out['actions'][-1]['action'] == 'healthy' and (not m_start.called))
out, m_start, _ = run_step([A, B], [dep('dep-1', 'acct-a', uptime=keepalive.KAGGLE_LIMIT_S - 100)], recovery=False)
acts = [a.get('action') for a in out['actions']]
rot = [a for a in out['actions'] if a.get('action') == 'preempt_rotate']
check('near-cap kernel triggers preempt_rotate', len(rot) == 1)
check('rotation goes to a different account', rot and rot[0]['to_account'] == 'b')
A2 = acct('acct-a', 'a', 20.0)
B2 = acct('acct-b', 'b', 0.1)
out, _, _ = run_step([A2, B2], [dep('dep-1', 'acct-a', uptime=keepalive.KAGGLE_LIMIT_S - 100)], recovery=False)
rot = [a for a in out['actions'] if a.get('action') == 'preempt_rotate']
check('falls back to same account when others drained', rot and rot[0]['to_account'] == 'a')
A3 = acct('acct-a', 'a', 0.2)
B3 = acct('acct-b', 'b', 0.0)
out, m_start, _ = run_step([A3, B3], [], recovery=True)
check('all drained -> drained, no start', any((a.get('action') == 'drained' for a in out['actions'])) and (not m_start.called))
d_old = dep('dep-old', 'acct-a', started=100.0)
d_new = dep('dep-new', 'acct-b', started=200.0)
out, _, m_stop = run_step([A, B], [d_new, d_old], recovery=False)
check('overlap retires oldest', any((a.get('action') == 'retire_extra' and a.get('stopped') == 'dep-old' for a in out['actions'])))
out, m_start, _ = run_step([A, B], [dep('dep-1', 'acct-a', status='STARTING', endpoint='')], recovery=True)
check('booting means wait', out['actions'][0]['action'] == 'wait' and (not m_start.called))
d_down = dep('dep-1', 'acct-a', status='ERROR')
d_down['last_update'] = time.time()
with mock.patch.object(sm, 'stop', lambda dep_id=None: {'ok': True}):
    out, m_start, _ = run_step([A, B], [d_down], recovery=False)
check('recent error -> cleared then wait', not m_start.called)
print()
if all((ok for _, ok in PASS)):
    print('ALL TESTS PASSED')
else:
    print('SOME TESTS FAILED')
    raise SystemExit(1)
