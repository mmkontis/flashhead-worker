import time, os, sys, shutil, subprocess
def log(m):
    print(f"[handler] {time.strftime('%H:%M:%S')} {m}", flush=True)
    try:
        with open("/runpod-volume/handler.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {m}\n")
    except Exception:
        pass
log("boot")
VOL_MODELS = "/runpod-volume/fh/models"
LOCAL_MODELS = "/models"
t0 = time.time()
if not os.path.isdir(LOCAL_MODELS):
    log("staging weights to local disk...")
    shutil.copytree(VOL_MODELS, LOCAL_MODELS)
    log(f"staged in {time.time()-t0:.0f}s")
sys.path.insert(0, "/app/SoulX-FlashHead")
os.chdir("/app/SoulX-FlashHead")
import runpod, base64, tempfile, argparse, io, contextlib
import torch
log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
import generate_video as gv
log("generate_video imported")
_cache = {}
_orig_get = gv.get_pipeline
def _cached(*a, **kw):
    mt = kw.get("model_type") or "lite"
    if mt not in _cache:
        t = time.time()
        _cache[mt] = _orig_get(*a, **kw)
        _cache[mt + "_load_s"] = round(time.time() - t, 1)
        log(f"pipeline {mt} loaded in {_cache[mt+'_load_s']}s")
    return _cache[mt]
gv.get_pipeline = _cached
# eager-load lite so FlashBoot snapshots a hot worker
try:
    _cached(world_size=1, ckpt_dir=LOCAL_MODELS + "/SoulX-FlashHead-1_3B",
            wav2vec_dir=LOCAL_MODELS + "/wav2vec2-base-960h", model_type="lite")
    d = torch.zeros(1, device="cuda"); del d
    log("eager lite load complete — worker hot")
except Exception as e:
    log(f"eager load failed (lazy fallback): {e}")
def handler(job):
    i = job["input"]
    if i.get("ping"):
        return {"pong": time.time(), "hot": "lite" in _cache}
    mt = i.get("model_type", "lite")
    log(f"job model={mt}")
    tmp = tempfile.mkdtemp()
    img, aud, out = tmp + "/c.png", tmp + "/a.wav", tmp + "/o.mp4"
    open(img, "wb").write(base64.b64decode(i["image_b64"]))
    open(aud, "wb").write(base64.b64decode(i["audio_b64"]))
    args = argparse.Namespace(ckpt_dir=LOCAL_MODELS+"/SoulX-FlashHead-1_3B", wav2vec_dir=LOCAL_MODELS+"/wav2vec2-base-960h",
        model_type=mt, save_file=out, base_seed=int(i.get("seed", 42)), cond_image=img, cond_image_dir=None,
        audio_path=aud, audio_encode_mode=i.get("audio_encode_mode", "once"), use_face_crop=False)
    t0 = time.time(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gv.generate(args)
    gen_s = round(time.time() - t0, 2)
    log(f"gen done {gen_s}s")
    steps = [float(l.rsplit(":", 1)[-1].strip().rstrip("s")) for l in buf.getvalue().splitlines() if "denoise per step" in l]
    return {"model_type": mt, "load_s": _cache.get(mt + "_load_s", 0), "gen_s": gen_s,
            "denoise_step_ms": round(sum(steps)/len(steps)*1000, 1) if steps else None,
            "video_b64": base64.b64encode(open(out, "rb").read()).decode()}
runpod.serverless.start({"handler": handler})
