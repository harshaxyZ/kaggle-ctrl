"""Generate a generic GPU notebook for an explicitly configured GGUF model."""
import json
import re
from settings import SETTINGS

def generate_kernel(spec, topic, api_key):
    if not spec.hf_repo or not spec.filename:
        raise ValueError('Configure hf_repo and filename in local settings before starting a server')
    if not re.fullmatch(r'[A-Za-z0-9._/-]+', spec.hf_repo) or not re.fullmatch(r'[A-Za-z0-9._/-]+', spec.filename):
        raise ValueError('Invalid model repository or filename')
    row = next(m for m in SETTINGS['models'] if m['key'] == spec.key)
    config = {'repo': spec.hf_repo, 'filename': spec.filename, 'projector': spec.mmproj,
              'revision': row.get('revision', 'main'), 'context': spec.ctx, 'parallel': spec.n_parallel,
              'alias': spec.key, 'extra_args': row.get('server_args', []),
              'session_seconds': SETTINGS.get('session_seconds', 11 * 3600)}
    # The event signing hook is injected by render_kernel before upload.
    return '''import os, sys, json, time, threading, subprocess, urllib.request, secrets
HF_REPO = ''' + repr(spec.hf_repo) + '''
API_KEY = "''' + api_key + '''"
TOPIC = ''' + repr(topic) + '''
CONFIG = json.loads(''' + repr(json.dumps(config)) + ''')
START = time.time()

def publish(stage, msg="", **extra):
    body = json.dumps({"stage": stage, "msg": msg, "ts": time.time(), **extra}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request("https://ntfy.sh/"+TOPIC, data=body, method="POST"), timeout=8).read()
    except Exception:
        pass

def log(*args):
    print("[serve]", *args, flush=True)

publish("boot", "kernel started")
os.makedirs("/kaggle/work", exist_ok=True)
subprocess.run([sys.executable,"-m","pip","install","-q","huggingface_hub"], check=True)
from huggingface_hub import hf_hub_download
publish("download", "fetching configured model")
model = hf_hub_download(CONFIG["repo"], CONFIG["filename"], revision=CONFIG["revision"], local_dir="/kaggle/work")
projector = hf_hub_download(CONFIG["repo"], CONFIG["projector"], revision=CONFIG["revision"], local_dir="/kaggle/work") if CONFIG["projector"] else ""
publish("install", "installing model server")
binary = install_llama_server(log=log, publish=publish)
if not binary:
    raise RuntimeError("Model server installation failed")
cmd = [binary,"-m",model,"--host","0.0.0.0","--port","8000","--ctx-size",str(CONFIG["context"]),"-ngl","999","--parallel",str(CONFIG["parallel"]),"--api-key",API_KEY,"--alias",CONFIG["alias"],"--metrics"]
if projector:
    cmd += ["--mmproj",projector]
cmd += CONFIG["extra_args"]
server = subprocess.Popen(cmd, stdout=open("/tmp/model-server.log","w"), stderr=subprocess.STDOUT)
for _ in range(480):
    if server.poll() is not None:
        raise RuntimeError("Model server exited; inspect notebook logs")
    try:
        response = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/models",headers={"Authorization":"Bearer "+API_KEY}),timeout=5)
        if response.status == 200:
            break
    except Exception:
        time.sleep(5)
else:
    raise RuntimeError("Model loading timed out")
publish("ready", "model loaded")
subprocess.run(["curl","-fsSL","https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64","-o","/tmp/cloudflared"],check=True)
os.chmod("/tmp/cloudflared",0o700)
tunnel = subprocess.Popen(["/tmp/cloudflared","tunnel","--url","http://127.0.0.1:8000","--no-autoupdate"],stdout=open("/tmp/model-tunnel.log","w"),stderr=subprocess.STDOUT)
import re
endpoint = ""
for _ in range(90):
    text = open("/tmp/model-tunnel.log").read()
    urls = re.findall(r"https://[a-z0-9-]+\\.trycloudflare\\.com",text)
    if urls:
        endpoint = urls[-1]
        break
    time.sleep(2)
if not endpoint:
    raise RuntimeError("Tunnel did not become ready")
publish("live", "endpoint is live", endpoint=endpoint, model=CONFIG["alias"], started=START)
while time.time()-START < CONFIG["session_seconds"] and server.poll() is None and tunnel.poll() is None:
    time.sleep(30)
    publish("heartbeat", "serving")
server.terminate()
tunnel.terminate()
'''
