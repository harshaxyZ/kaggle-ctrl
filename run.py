"""Entry point: python run.py -> http://127.0.0.1:8787

Single-instance guard. Two dashboards on the same port would fight over
server_state.json and over the serving kernels: each keeps its own idea of
which deployment is live, so one of them keeps cancelling the other's kernels
and the router flips between endpoints. The lock makes that impossible.
"""
import os
import sys
import socket
from dotenv import load_dotenv
load_dotenv(__import__("pathlib").Path(__file__).resolve().parent / ".env", override=False)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if not os.environ.get('KAGGLE_API_TOKEN'):
    for p in (os.path.expanduser('~/.kaggle/access_token'),):
        if os.path.exists(p):
            os.environ['KAGGLE_API_TOKEN'] = open(p, encoding='utf-8').read().strip()
            break
PORT = int(os.environ.get('PORT', '8787'))
HOST = os.environ.get('HOST', '127.0.0.1')

def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0

def _kill_existing() -> None:
    """Kill a previous dashboard holding this port, so a restart is clean."""
    if os.name == 'nt':
        os.system('for /f "tokens=5" %%p in (\'netstat -ano -p TCP ^| findstr ":%d LISTENING"\') do taskkill /F /PID %%p >nul 2>&1' % port)
    else:
        self_pid = os.getpid()
        os.system(f"""for p in $(pgrep -f 'python.*run.py'); do [ "$p" -eq {self_pid} ] || kill "$p"; done 2>/dev/null || true""")
    import time
    time.sleep(2)
if _port_in_use(HOST, PORT):
    print(f'Port {PORT} is busy - stopping the old dashboard first.')
    _kill_existing()
    if _port_in_use(HOST, PORT):
        print(f'FATAL: port {PORT} still in use. Close it manually and retry.')
        sys.exit(1)
import uvicorn
if __name__ == '__main__':
    print(f'Kaggle dashboard -> http://{HOST}:{PORT}')
    uvicorn.run('app:app', host=HOST, port=PORT, log_level='info')
