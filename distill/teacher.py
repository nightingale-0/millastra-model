"""Teacher LLM client: DeepSeek API (default) or a local llama.cpp server, both OpenAI-compatible.

    TEACHER=deepseek  (default)  model deepseek-flash, thinking disabled; key read from
                                  deepseek.env (API_KEY=...) at runtime, never printed
    TEACHER=local                 tools/llama/llama-server.exe -m <gguf> -ngl 99 -c 16384 -np 8 --jinja --port 8080

run_tasks() sends many requests concurrently and appends one JSON line per finished
task to an output file; tasks already in the file are skipped (resumable). Token usage
and an estimated DeepSeek cost are logged.
"""
import json, os, re, threading, time, urllib.request, urllib.error, concurrent.futures as cf

BACKEND = os.environ.get("TEACHER", "deepseek")
LOCAL_URL = os.environ.get("TEACHER_URL", "http://127.0.0.1:8080")
DS_URL, DS_MODEL = "https://api.deepseek.com/chat/completions", "deepseek-flash"
# yuan per 1M tokens, off-peak (peak = x2): input cache hit / miss, output
DS_PRICE = {"hit": 0.02, "miss": 1.0, "out": 4.0}
WORKERS = 32 if BACKEND == "deepseek" else 8

_key = None
def _api_key():
    global _key
    if _key is None:
        path = os.path.join(os.path.dirname(__file__), "..", "deepseek.env")
        with open(path, encoding="utf-8") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                if k.strip() in ("API_KEY", "export API_KEY"): _key = v.strip().strip('"').strip("'")
        if not _key: raise RuntimeError("API_KEY not found in deepseek.env")
    return _key

class FatalAPIError(RuntimeError):
    """401/402/403: abort the whole run instead of writing a partial dataset."""

usage_total = {"prompt": 0, "hit": 0, "out": 0}
_ulock = threading.Lock()

def cost_yuan(u=None):
    u = u or usage_total
    return ((u["prompt"] - u["hit"]) * DS_PRICE["miss"] + u["hit"] * DS_PRICE["hit"] + u["out"] * DS_PRICE["out"]) / 1e6

def chat(messages, max_tokens=1024, temperature=0.7, top_p=0.8, presence_penalty=1.0, timeout=600):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "presence_penalty": presence_penalty}
    headers = {"Content-Type": "application/json"}
    if BACKEND == "deepseek":
        body.update(model=DS_MODEL, thinking={"type": "disabled"})
        headers["Authorization"] = "Bearer " + _api_key()
        url = DS_URL
    else:
        body.update(top_p=top_p, chat_template_kwargs={"enable_thinking": False})
        url = LOCAL_URL + "/v1/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(2 ** attempt); continue
            if e.code in (401, 402, 403):        # bad key / no balance: retrying cannot help
                raise FatalAPIError(f"HTTP {e.code}: {e.read()[:200]!r}") from None
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:   # dropped connections
            if attempt < 5:
                time.sleep(2 ** attempt); continue
            raise RuntimeError(f"network: {e}") from None
    msg = out["choices"][0]["message"].get("content") or ""
    u = out.get("usage", {})
    with _ulock:
        usage_total["prompt"] += u.get("prompt_tokens", 0)
        usage_total["hit"] += u.get("prompt_cache_hit_tokens", 0)
        usage_total["out"] += u.get("completion_tokens", 0)
    return re.sub(r"<think>.*?</think>", "", msg, flags=re.S).strip(), u

def wait_ready(timeout=600):
    if BACKEND == "deepseek": return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(LOCAL_URL + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok": return True
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("teacher server not ready")

def parse_json(text):
    """first JSON array/object in a reply (tolerates code fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("[", "]"), ("{", "}")):
        a, b = text.find(opener), text.rfind(closer)
        if a >= 0 and b > a:
            try: return json.loads(text[a:b + 1])
            except Exception: pass
    # one JSON value per line (e.g. ["a"]\n["b"] or {..}\n{..}): merge into one list
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line: continue
        try: v = json.loads(line)
        except Exception: continue
        out.extend(v if isinstance(v, list) else [v])
    return out or None

def run_tasks(tasks, handle, out_path, workers=None, log_every=50):
    """tasks: list of dicts with unique 'id'; handle(task) -> dict or None (written with the id)."""
    workers = workers or WORKERS
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try: done.add(json.loads(line)["id"])
                except Exception: pass
    todo = [t for t in tasks if t["id"] not in done]
    print(f"{out_path}: {len(done)} done, {len(todo)} to go ({BACKEND}, {workers} workers)", flush=True)
    lock, n, failed, t0 = threading.Lock(), 0, 0, time.time()
    out0 = dict(usage_total)
    fatal = []
    def one(t):
        err = None
        if fatal: return None
        for attempt in range(3):
            try: return handle(t)
            except FatalAPIError as e:
                fatal.append(e); return None
            except Exception as e:
                err = e; time.sleep(2)
        print(f"task {t['id']} failed: {str(err)[:200]}", flush=True)
        return None
    with open(out_path, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(one, todo):
            if res is None:
                failed += 1; continue
            with lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
                n += 1
                if n % log_every == 0 or n == len(todo):
                    dt = time.time() - t0
                    gen = usage_total["out"] - out0["out"]
                    print(f"  {n}/{len(todo)}  {gen/dt:.0f} out tok/s  eta {(len(todo)-n)*dt/max(n,1)/60:.1f} min"
                          f"  tokens in {usage_total['prompt']/1e6:.2f}M (hit {usage_total['hit']/1e6:.2f}M) out {usage_total['out']/1e6:.2f}M"
                          + (f"  ~{cost_yuan():.2f} yuan off-peak" if BACKEND == "deepseek" else ""), flush=True)
    print(f"{out_path}: finished {n}, failed {failed}", flush=True)
    if fatal: raise SystemExit(f"aborted: {fatal[0]}")
    if failed > max(5, len(todo) // 20):          # >5% failures: do not continue on a partial set
        raise SystemExit(f"aborted: {failed} of {len(todo)} tasks failed (rerun to resume)")
