"""Streaming TPS/TPM benchmark for the kaggle-ctrl router or a direct endpoint.

Measures what actually matters:
  - TTFT (time to first token)
  - decode tok/s per stream (chunks after the first / decode time)
  - aggregate tok/s and TPM across parallel clients

Usage:
  python bench.py --base http://127.0.0.1:8787/v1 --key sk-kctrl-...       --tokens 256 --requests 4 --parallel 1
"""
from __future__ import annotations
import argparse
import json
import statistics
import threading
import time
import urllib.request
PROMPT = 'Write a detailed technical explanation of how speculative decoding works in modern LLM inference, including draft models, verification passes, and acceptance rates. Cover the math.'

def one_request(base: str, key: str, model: str, max_tokens: int, prompt: str) -> dict:
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens, 'temperature': 0.7, 'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    t0 = time.time()
    ttft = None
    chunks = 0
    usage = {}
    timings = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get('usage'):
                usage = obj['usage']
            if obj.get('timings'):
                timings = obj['timings']
            for ch in obj.get('choices', []):
                if ch.get('timings'):
                    timings = ch['timings']
                delta = ch.get('delta', {})
                c = delta.get('content') or delta.get('reasoning_content')
                if c:
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        chunks += 1
    t_end = time.time()
    out_tok = usage.get('completion_tokens') or chunks + (1 if ttft is not None else 0)
    decode_s = t_end - t0 - ttft if ttft is not None else 0.0
    res = {'ok': ttft is not None, 'ttft_s': round(ttft or 0, 3), 'total_s': round(t_end - t0, 3), 'out_tokens': out_tok, 'decode_s': round(decode_s, 3), 'decode_tps': round(out_tok / decode_s, 2) if decode_s > 0 else 0.0, 'in_tokens': usage.get('prompt_tokens', 0)}
    if timings:
        res['server_decode_tps'] = round(timings.get('predicted_per_second', 0), 2)
        res['server_prompt_tps'] = round(timings.get('prompt_per_second', 0), 2)
        res['draft_accept_pct'] = round(100.0 * timings.get('draft_n_accepted', 0) / timings['draft_n'], 1) if timings.get('draft_n') else None
    return res

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--model', default=__import__('settings').SETTINGS['default_model'])
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--requests', type=int, default=4)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--prompt', default=PROMPT)
    a = ap.parse_args()
    results: list[dict] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        for _ in range(n):
            r = one_request(a.base, a.key, a.model, a.tokens, a.prompt)
            with lock:
                results.append(r)
                print('  req:', json.dumps(r), flush=True)
    per_worker = a.requests // a.parallel
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(per_worker,)) for _ in range(a.parallel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    oks = [r for r in results if r['ok']]
    print('\n=== SUMMARY ===')
    print(f'requests ok: {len(oks)}/{len(results)}  parallel={a.parallel}  wall={wall:.1f}s')
    if oks:
        med_tps = statistics.median((r['decode_tps'] for r in oks))
        mean_ttft = statistics.mean((r['ttft_s'] for r in oks))
        total_out = sum((r['out_tokens'] for r in oks))
        total_in = sum((r['in_tokens'] for r in oks))
        print(f'median client decode TPS: {med_tps:.2f}')
        srv = [r['server_decode_tps'] for r in oks if r.get('server_decode_tps')]
        if srv:
            print(f'median SERVER decode TPS: {statistics.median(srv):.2f}  (llama.cpp timings)')
        acc = [r['draft_accept_pct'] for r in oks if r.get('draft_accept_pct') is not None]
        if acc:
            print(f'MTP draft acceptance: {statistics.mean(acc):.0f}%')
        print(f'mean TTFT: {mean_ttft:.2f}s')
        print(f'aggregate: {total_out / wall:.2f} tok/s out  |  TPM out: {total_out / wall * 60:.0f}  |  in+out TPM: {(total_in + total_out) / wall * 60:.0f}')
if __name__ == '__main__':
    main()
