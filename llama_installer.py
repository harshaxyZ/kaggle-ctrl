"""
llama_installer.py - the install routine used by the serving kernel.

Kept as its own module (imported and called by the generated kernel) so it can
be unit-tested locally instead of being retyped inside an f-string template,
which is where the escaping bugs kept coming from.

What it does, and why:
  * Downloads a prebuilt CUDA llama.cpp release. No 20 minute source build.
  * The plain archive holds the binaries; the cudart archive holds only the
    CUDA runtime. Kaggle's image has no libcudart.so.12, and llama-server
    will not even load without it, so both are needed.
  * The archive has a top-level directory named after the build tag, so one
    leading path component is stripped.
  * The binary is located with a directory walk rather than a hardcoded path.
  * The binary is executed once to confirm it runs before it is relied on.
  * If none of that works, fall back to building from source.
"""
from __future__ import annotations
import os
import subprocess
import time
LLAMA_DIR = '/kaggle/llama'
LLAMA_BUILDS = ['b12438', 'b12436', 'b12435', 'b12428']

def _sh(cmd: str, timeout: int=900):
    return subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=timeout)

def _find_llama_server() -> str:
    for root, _dirs, files in os.walk(LLAMA_DIR):
        for f in files:
            if f == 'llama-server':
                p = os.path.join(root, f)
                if os.access(p, os.X_OK):
                    return p
    return ''

def _lib_dirs() -> str:
    dirs = set()
    for root, _dirs, files in os.walk(LLAMA_DIR):
        for f in files:
            if f.endswith('.so') or '.so.' in f:
                dirs.add(root)
    return ':'.join(sorted(dirs))

def _download(url: str, dest: str, timeout: int=1000) -> bool:
    r = _sh("curl -fsSL --retry 3 --retry-delay 3 --max-time 1200 -o '%s' '%s'" % (dest, url), timeout=timeout)
    return r.returncode == 0 and os.path.exists(dest)

def install_llama_server(log=print, publish=None) -> str:
    """Install llama-server. Returns its path, or an empty string on failure."""

    def say(stage, msg, **extra):
        log(stage, msg)
        if publish:
            try:
                publish(stage, msg, **extra)
            except Exception:
                pass
    os.makedirs(LLAMA_DIR, exist_ok=True)
    for build in LLAMA_BUILDS:
        plain_url = 'https://github.com/ggml-org/llama.cpp/releases/download/' + build + '/llama-' + build + '-bin-ubuntu-cuda-12.8-x64.tar.gz'
        cud_url = 'https://github.com/ggml-org/llama.cpp/releases/download/' + build + '/cudart-llama-' + build + '-bin-ubuntu-cuda-12.8-x64.tar.gz'
        say('install', 'fetching llama.cpp ' + build)
        if not _download(plain_url, '/tmp/llama.tar.gz'):
            say('install', 'download failed ' + build)
            continue
        size = os.path.getsize('/tmp/llama.tar.gz')
        say('install', 'archive bytes %d' % size)
        if size < 50000000:
            say('install', 'archive too small')
            continue
        _sh('cd %s && rm -rf bin lib build cudart && tar -xzf /tmp/llama.tar.gz --strip-components=1' % LLAMA_DIR, timeout=600)
        binp = _find_llama_server()
        say('install', 'binary: %s' % (binp or 'NOT FOUND'))
        if not binp:
            continue
        os.chmod(binp, 493)
        say('install', 'fetching cuda runtime for ' + build)
        _sh("mkdir -p %s/cudart && curl -fsSL --retry 2 --max-time 900 -o /tmp/cudart.tar.gz '%s' && tar -xzf /tmp/cudart.tar.gz -C %s/cudart --strip-components=1" % (LLAMA_DIR, cud_url, LLAMA_DIR), timeout=1000)
        libpath = _lib_dirs()
        say('install', 'libdirs: ' + libpath[:300])
        env = {'LD_LIBRARY_PATH': libpath + ':' + os.environ.get('LD_LIBRARY_PATH', '')}
        v = subprocess.run([binp, '--version'], capture_output=True, text=True, env={**os.environ, **env}, timeout=120)
        out = (v.stdout or '') + (v.stderr or '')
        say('install', 'version rc=%d %s' % (v.returncode, out.strip()[:200]))
        lowered = out.lower()
        looks_right = v.returncode == 0 and ('version' in lowered or 'llama' in lowered)
        if looks_right:
            os.environ['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH']
            say('install_ok', 'llama.cpp ready', build=build, path=binp)
            return binp
    say('install', 'building llama.cpp from source')
    _sh('rm -rf %s && git clone --depth 1 https://github.com/ggml-org/llama.cpp %s' % (LLAMA_DIR, LLAMA_DIR), timeout=900)
    _sh('cd %s && cmake -B build -DGGML_CUDA=on -DLLAMA_CURL=off -DLLAMA_BUILD_TESTS=off -DLLAMA_BUILD_EXAMPLES=off >/dev/null 2>&1 && cmake --build build --config Release -j4 --target llama-server >/dev/null 2>&1' % LLAMA_DIR, timeout=3000)
    binp = _find_llama_server()
    if binp:
        os.chmod(binp, 493)
        say('install_ok', 'llama.cpp built', path=binp)
        return binp
    say('error', 'could not obtain llama-server')
    return ''

def preload_cuda(log=print) -> str:
    """Make the CUDA runtime loadable by the current process.

    Kaggle's image does not put libcudart on the loader path, so the
    llama-cpp-python wheel fails with "libcudart.so.12: cannot open shared
    object file". dlopen the runtime ourselves with RTLD_GLOBAL so the
    symbols are resident before llama_cpp is imported.

    Returns the directory list that should be on LD_LIBRARY_PATH for child
    processes, which start with a fresh loader and do not inherit the dlopen.
    """
    import ctypes
    import glob
    import site
    for pkg in ('nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12', 'nvidia-cuda-nvrtc-cu12'):
        _sh('pip install -q --no-cache-dir %s >/dev/null 2>&1' % pkg, timeout=600)
    search = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        search += glob.glob(os.path.join(sp, 'nvidia', '*', 'lib'))
    search += glob.glob('/usr/local/cuda*/lib64')
    for extra in ('/usr/local/lib', '/usr/lib/x86_64-linux-gnu'):
        search.append(extra)
    loaded = []
    for name in ('libcudart.so*', 'libcublas.so*', 'libcublasLt.so*', 'libnvrtc.so*'):
        for d in search:
            hits = sorted(glob.glob(os.path.join(d, name)))
            for h in hits:
                try:
                    ctypes.CDLL(h, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(h)
                    break
                except OSError:
                    continue
    log('cuda libs preloaded', len(loaded))
    dirs = sorted({os.path.dirname(h) for h in loaded})
    return ':'.join(dirs)
if __name__ == '__main__':
    p = install_llama_server(log=lambda *a: print(*a, flush=True))
    print('RESULT', p)
