"""
tunnel.py - publish the local dashboard to the internet, for a remote bot.

Kaggle kernels have no inbound ports, so a model running there can only be
reached through an outbound tunnel. The same is true of this dashboard if the
remote client does not run on this machine.

So: the dashboard dials out to Cloudflare, Cloudflare dials back, and the
resulting public URL is published two ways:
  * a local file (dashboard_url.txt) for anything on this machine
  * an ntfy topic (per-install discovery topic) for anything else, which can poll for
    the current URL without any credentials

The URL is not stable across restarts with a quick tunnel, so a remote client
should always re-read it rather than cache it. If a stable hostname is
needed, point CLOUDFLARED at a named tunnel instead (see start_named).
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
URL_FILE = HERE / 'dashboard_url.txt'
NTFY_TOPIC = 'kgctrl-' + __import__('settings').installation_id()
_LOCK = threading.RLock()
_proc: subprocess.Popen | None = None
_url: str = ''
_started_at: float = 0.0

def _download_cloudflared() -> str:
    """Return a path to a usable cloudflared binary, downloading it if needed."""
    exe = shutil.which('cloudflared')
    if exe:
        return exe
    local = HERE / 'bin' / ('cloudflared.exe' if os.name == 'nt' else 'cloudflared')
    if local.exists():
        return str(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
    else:
        url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'
    try:
        urllib.request.urlretrieve(url, str(local))
        os.chmod(local, 493)
        return str(local)
    except Exception:
        return ''

def _publish(url: str) -> None:
    try:
        URL_FILE.write_text(url, encoding='utf-8')
    except Exception:
        pass
    try:
        body = json.dumps({'url': url, 'ts': time.time()}).encode()
        req = urllib.request.Request('https://ntfy.sh/' + NTFY_TOPIC, data=body, method='POST', headers={'Title': 'dashboard', 'Tags': 'globe_with_meridians'})
        urllib.request.urlopen(req, timeout=6).read()
    except Exception:
        pass

def start(port: int=8787) -> dict:
    """Start a quick tunnel for the dashboard. Returns the public URL."""
    global _proc, _url, _started_at
    with _LOCK:
        if _proc and _proc.poll() is None and _url:
            return {'ok': True, 'url': _url, 'already': True}
        exe = _download_cloudflared()
        if not exe:
            return {'ok': False, 'message': 'cloudflared unavailable'}
        _proc = subprocess.Popen([exe, 'tunnel', '--url', f'http://127.0.0.1:{port}', '--no-autoupdate'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        url = ''
        deadline = time.time() + 60
        for line in _proc.stdout:
            m = re.search('https://[a-z0-9\\-]+\\.trycloudflare\\.com', line)
            if m:
                url = m.group(0)
                break
            if time.time() > deadline:
                break
        if not url:
            return {'ok': False, 'message': 'tunnel did not report a url'}
        _url = url
        _started_at = time.time()
        _publish(url)
        threading.Thread(target=_drain, args=(_proc,), daemon=True).start()
        return {'ok': True, 'url': url}

def _drain(proc: subprocess.Popen) -> None:
    try:
        for _ in proc.stdout:
            pass
    except Exception:
        pass

def stop() -> dict:
    global _proc, _url, _started_at
    with _LOCK:
        if _proc and _proc.poll() is None:
            try:
                _proc.terminate()
            except Exception:
                pass
        _proc = None
        _url = ''
        _started_at = 0.0
        try:
            if URL_FILE.exists():
                URL_FILE.unlink()
        except Exception:
            pass
    return {'ok': True, 'message': 'tunnel stopped'}

def status() -> dict:
    global _url, _started_at
    with _LOCK:
        alive = bool(_proc and _proc.poll() is None)
        return {'running': alive and bool(_url), 'url': _url, 'uptime_s': int(time.time() - _started_at) if _started_at else 0, 'topic': NTFY_TOPIC, 'file': str(URL_FILE)}
