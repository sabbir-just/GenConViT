%%writefile /kaggle/working/genconvit_tfkd.py
# =====================================================================================
#  GenConViT-TFKD  —  Lightweight GenConViT-based backbone + Spatial-Artifact + Frequency + Temporal + KD
#  ONE-CELL Kaggle script (train / evaluate).  Edit ONLY the CONFIG block (and build_teacher_model()/teacher_forward()).
#  NOTE: the backbone is a LIGHTWEIGHT GenConViT-BASED design (AE/VAE + ConvNeXt-T/Swin-T).
#        It is NOT an exact reproduction of the original 695M-parameter GenConViT implementation.
#        Differences: AE/VAE branches are trained JOINTLY end-to-end (not independently pretrained) and ONE ConvNeXt/Swin weight set is SHARED across original + reconstructed inputs.
#        A lightweight generative-INPUT warm-up (GENERATIVE_INPUT_WARMUP_EPOCHS) fades the reconstructions into the backbone input so the pretrained backbone never sees random reconstructions.
# =====================================================================================
import os, sys, json, time, math, random, copy, re, io, hashlib, platform, warnings, subprocess, shutil, pickle, gc
os.environ["NCCL_P2P_DISABLE"] = "1"                                  # avoids the 2xT4 DataParallel/NCCL hang (must be set before importing torch)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"     # less allocator fragmentation on 15 GB T4s
import urllib.request
from collections import defaultdict
from types import SimpleNamespace
SESSION_START = time.time()
warnings.simplefilter("default")                                                   # real warnings are visible again
warnings.filterwarnings("ignore", category=ResourceWarning)                        # unclosed-file noise from notebook I/O helpers (harmless)
warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*")          # jupyter_client deprecation, not our code
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

def _ensure(pkg, imp=None):
    try: __import__(imp or pkg)
    except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
_ensure("timm"); _ensure("scikit-learn", "sklearn")

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef, log_loss)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import zipfile
FACES_ZIP = "/kaggle/input/notebooks/hafsatalukder/genconvit-1/_output_.zip"
FACES_DIR = "/tmp/faces"                                   # local disk: not counted in the 19.5 GiB output quota

def ensure_faces():
    """Extract ffpp_faces/ and cdf_faces/ from the saved crop-notebook output (once per session; skipped if already extracted)."""
    flag = os.path.join(FACES_DIR, ".extracted_ok")
    if os.path.exists(flag): print("faces already extracted:", FACES_DIR); return
    direct = os.path.dirname(FACES_ZIP)                    # if Kaggle exposes the folder unzipped, no extraction is needed
    if os.path.isdir(os.path.join(direct, "ffpp_faces")): raise RuntimeError(f"Output is already unzipped at {direct}: point FFPP_PATH/CELEBDF_PATH there and remove ensure_faces().")
    if not os.path.exists(FACES_ZIP): raise FileNotFoundError(f"Crop output zip not found: {FACES_ZIP} (is the crop notebook added as an input?)")
    os.makedirs(FACES_DIR, exist_ok=True)
    with zipfile.ZipFile(FACES_ZIP) as z:
        allnames = z.namelist(); names = [n for n in allnames if n.startswith(("ffpp_faces/", "cdf_faces/"))]
        if not names: raise RuntimeError(f"No ffpp_faces/ or cdf_faces/ inside the zip. Top-level entries: {sorted({n.split('/')[0] for n in allnames})[:20]}")
        print(f"extracting {len(names):,} files to {FACES_DIR} ...", flush=True)
        for n in tqdm(names, desc="unzip"): z.extract(n, FACES_DIR)
    open(flag, "w").close()

def check_faces():
    """Faces are a Kaggle Dataset (already unzipped): just verify the folders exist."""
    for k in ("FFPP_PATH", "CELEBDF_PATH"):
        p = CONFIG[k]
        if not os.path.isdir(p): raise FileNotFoundError(f"{k} not found: {p}  (is the 'the-faces' dataset added as an input?)")
        print(f"{k}: {p} -> {sorted(os.listdir(p))[:8]}", flush=True)

GENCONVIT_NOTE = "lightweight GenConViT-based backbone; not an exact reproduction of the original 695M-parameter GenConViT implementation"
GENCONVIT_DETAIL = ("lightweight GenConViT-based backbone (AE/VAE + ConvNeXt-T/Swin-T); AE/VAE trained jointly end-to-end with a generative-input warm-up "
                    "(no separate pretraining); backbone weights shared across original and reconstructed inputs; original GenConViT reports ~695M parameters")

# =====================================================================================
#                                     CONFIG
# =====================================================================================
CONFIG = {
    # ---------------- run control ----------------
    "MODE": "evaluate",                    # "train" | "evaluate"
    "EXP_NAME": "abl_T1F1A1KD1",                   # "auto" -> genconvit_T{0/1}F{0/1}A{0/1}KD{0/1}
    "OUTPUT_DIR": "/kaggle/working/output",
    "RESUME_CHECKPOINT": "",              # e.g. /kaggle/input/<prev-output>/output/<exp>/checkpoints/session_end_epoch005.pth
    "AUTO_RESUME": True,                  # also resume from OUTPUT_DIR/<exp>/checkpoints/last.pth if it exists
    "BEST_CHECKPOINT": "/kaggle/input/notebooks/hafsatalukder/genconvit-1/output/abl_T1F1A1KD0_SBI/checkpoints/best_auc.pth",                # used by MODE="evaluate" (best_auc.pth)
    "EVAL_WEIGHTS": "best",               # "best" | "raw" | "ema"

    # ---------------- data (paths unchanged) ----------------
    "FFPP_PATH": "/kaggle/input/datasets/hafsatalukder/the-faces/ffpp_faces",
    "CELEBDF_PATH": "/kaggle/input/datasets/hafsatalukder/the-faces/cdf_faces",
    
    "DFDC_PATH": "/kaggle/input/datasets/itamargr/dfdc-faces-of-the-train-sample/validation",
    "DIFFUSION_PATH": "/kaggle/input/datasets/syedazmulhasansabbir/diffusion/diffusion",   # OPTIONAL ("" to skip; skipped with a message if missing)
    "DIFFUSION_MAX_PER_FOLDER": 2000,
    # ---- cached file listing (same pickle as multi_teacher_kd_v8; only FF++/CDF/Diffusion match, DFDC falls back to a live scan) ----
    "USE_SCAN_CACHE": False,              # cache lists the OLD uncropped paths; live os.walk on the cropped copies
    "SCAN_CACHE_PATH": "/kaggle/input/datasets/asmabagum/dataset-scan/dataset_scans_v40.pkl",
    "SCAN_CACHE_VERIFY_N": 200,           # number of cached paths probed with os.path.exists
    "MAX_EVAL_VIDEOS": None,              # cap per cross-dataset (None = all)
    # ================= MODIFIED =================
    # frame -> video grouping: "auto" | "video_dirs" | "filename_prefix" | "single_image"
    "FFPP_GROUPING_MODE": "auto",
    "CELEBDF_GROUPING_MODE": "auto",
    "DFDC_GROUPING_MODE": "auto",
    # official FF++ split
    "FFPP_SPLIT_DIR": "",                 # dir with official train.json / val.json / test.json; if "" -> auto-download (Internet ON)
    "FFPP_SPLIT_URL_BASE": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits",
    "ALLOW_DEBUG_SPLIT": False,           # DEBUG_ONLY identity-inferred split. NEVER for the paper run.
    # several REAL entries with the same original-video id (e.g. two copies of the extracted frames): "keep_longest" | "keep_first" | "error"
    "REAL_DUPLICATE_POLICY": "match_fake_pipeline",
    "SPLIT_FRACTIONS": (0.72, 0.14, 0.14),# used only by the DEBUG_ONLY split

    # temporal sampling
    "TEMPORAL_SAMPLING": "window",        # "window" (uniform stride inside a window) | "contiguous" (stride forced to 1)
    "TRAIN_WINDOW": 36,                   # random temporal window (frames) for training clips
    "TEMPORAL_STRIDE": 2,                 # frame stride inside the window
    "EVAL_WINDOW": 36, "EVAL_WINDOW_STRIDE": 16,   # evaluation windows and hop
    "MAX_EVAL_CLIPS": 8,                  # max clips/video at evaluation (evenly spaced windows)

    # ---------------- teacher / KD ----------------
    "TEACHER_CHECKPOINT": "/kaggle/input/models/syedazmulhasansabbir/dp-vit-face-forensic/tensorflow2/default/1/dpvit_faceforensics_final.pth",
    "TEACHER_ARCH": "vit_base_patch16_224",   # only used by the DEFAULT build_teacher_model()
    "TEACHER_NUM_CLASSES": 2,
    "TEACHER_FAKE_INDEX": 1,                  # which logit is 'fake' (1 = [real, fake])
    "TEACHER_MEAN": (0.485, 0.456, 0.406), "TEACHER_STD": (0.229, 0.224, 0.225),
    "TEACHER_INPUT_SIZE": 224,
    "TEACHER_BGR": False,                     # test switch: feed the teacher BGR (cv2-style) instead of RGB
    "RUN_TEACHER_PREFLIGHT": True,            # full teacher FF++-val evaluation + label-order check (True for first run; optional later)

    # ---------------- ablation switches (one experiment per run) ----------------
    "USE_SPATIAL_ARTIFACT": True, "USE_FREQUENCY": True, "USE_TEMPORAL": True, "USE_KD": False,

    # ---------------- model ----------------
    "GENCONVIT_VARIANT": "both",          # "ed" (lighter) | "vae" | "both" (AE + VAE branches, as in GenConViT)
    "BACKBONES": ("convnext_tiny", "swin_tiny_patch4_window7_224"),
    "PRETRAINED": True,                   # needs Internet ON; FAILS LOUDLY otherwise
    "ALLOW_DEBUG_FALLBACK": False,        # DEBUG ONLY: permit random-init backbones if pretrained download fails
    "EMBED_DIM": 256, "TEMP_LAYERS": 2, "TEMP_HEADS": 4, "DROPOUT": 0.1,
    "DROP_PATH": 0.5,                     # stochastic depth in ConvNeXt/Swin (regularisation against overfitting)
    "IMAGE_SIZE": 224, "NUM_FRAMES": 12,
    "FREEZE_BACKBONE_EPOCHS": 2, "GRAD_CHECKPOINT": False,

    # ---------------- optimisation ----------------
    "SEED": 42, "BATCH_SIZE": 2, "TARGET_EFFECTIVE_BATCH": 16, "EVAL_BATCH_SIZE": 8,
    "NUM_EPOCHS": 14, "LR_BACKBONE": 2e-5, "LR_NEW_MODULES": 2e-4, "WEIGHT_DECAY": 0.05,
    "WARMUP_EPOCHS": 1, "MIN_LR_RATIO": 0.01, "GRAD_CLIP": 1.0, "LABEL_SMOOTHING": 0.02,
    "USE_AMP": True, "USE_EMA": True, "EMA_DECAY": 0.998, "USE_DATAPARALLEL": False,
    "NUM_WORKERS": 4, "TRAIN_CLIPS_PER_EPOCH": 3200, "EARLY_STOP_PATIENCE": 10,
    "AUG_BLUR_PROB": 0.10, "AUG_JPEG_PROB": 0.30, "AUG_DOWNSCALE_PROB": 0.10, "AUG_COLOR_PROB": 0.30, "AUG_NOISE_PROB": 0.05,
    "SBI_PROB": 0.0,
    "AUG_RAMP_EPOCHS": 3,                 # >0: augmentation probabilities ramp linearly 0 -> full over this many epochs (curriculum)
    "NORM_TYPE": "gn",                    # "gn" | "bn"  (bisection switch)                      # fraction of REAL training clips turned into self-blended pseudo-fakes (label 1). Try 0.3.

    # ---------------- loss weights ----------------
    "LAMBDA_CE": 1.0, "FRAME_CE_WEIGHT": 0.25, "LAMBDA_KD": 0.1, "TEMPERATURE": 3.0,
    "AUX_CE_WEIGHT": 0.05,                 # deep supervision: per-branch (spatial/artifact/frequency) frame classifiers
    "KD_TEACHER_CORRECT_ONLY": True,      # distill only where the teacher is correct
    "LAMBDA_TC": 0.01,                    # only active when USE_TEMPORAL=True
    "LAMBDA_REC": 1.0, "LAMBDA_KL": 1e-3, # GenConViT generative-branch terms (higher: keep the AE a real reconstructor)
    # ================= MODIFIED =================
    "GENERATIVE_WARMUP_EPOCHS": 0,        # L_REC / L_KL LOSS weights ramp linearly 0 -> LAMBDA_* over this many epochs (0 = off)
    # ================= MODIFIED =================
    "GENERATIVE_INPUT_WARMUP_EPOCHS": 0,
    "GEN_PRETRAIN_STEPS": 3000, "GEN_PRETRAIN_BATCH": 4, "GEN_PRETRAIN_LR": 1e-3,   # stage 0: reconstruction-only AE/VAE pretraining (as in GenConViT)

    # ---------------- session safety ----------------
    "MAX_SESSION_MINUTES": 645, "DIAG_RESERVE_MINUTES": 22, "VAL_TIME_GUESS_MINUTES": 6.0, "DIAG_FINAL_MARGIN_MINUTES": 4.0,
    "EMERGENCY_CKPT_MINUTES": 30, "MAX_EPOCHS_THIS_SESSION": 0, "KEEP_EPOCH_CKPTS": 2, "NAN_PATIENCE": 10,
    "VAL_CLIPS_PER_VIDEO": 1, "RUN_PREFLIGHT": True,

    # ---------------- evaluation ----------------
    "TTA_FLIP": False,                    # set True ONLY for MODE="evaluate"
    "VIDEO_AGGREGATION": "mean",         # mean | median | max | topk_mean  (fix BEFORE final evaluation; never tune on test sets)
    "BOOTSTRAP_ITERS": 1000,              # 500 is fine for preliminary runs
    "SAVE_VISUAL_ANALYSIS": False,

    # ================= ROUND-1 PATCH KEYS =================
    # Python keeps the LAST value of a duplicated dict key, so entries below override the same key higher up in CONFIG.
    # NUM_FRAMES / TRAIN_WINDOW / TEMPORAL_STRIDE are deliberately NOT overridden: your current values (T=12) stay.
    # validation (B2/B3)
    "EVAL_CLIP_MODE": "cover", "VAL_MODE": "full", "VAL_FAST_REAL": 100, "VAL_FAST_PER_GROUP": 40, "VAL_FAST_CLIPS": 4, "VAL_FULL_CLIPS": 4,
    "NUM_WORKERS_EVAL": 4, "VAL_AGG_SWEEP": True,
    # data / sampler (B4/D1)
    "REAL_DUPLICATE_POLICY": "match_fake_pipeline", "DUP_CONTENT_CHECK": True, "SAMPLER_MODE": "stratified",
    "REAL_FRAC": 0.5,                     # share of REAL clips per epoch in the stratified sampler (0.5 = original behaviour)
    # optimizer (C1)
    "LR_BACKBONE": 2e-5, "LR_GENERATIVE": 3e-4, "LR_FORENSIC": 3e-4, "LR_TEMPORAL": 2e-4, "LR_FUSION": 2e-4, "LR_HEADS": 5e-4, "FREEZE_BACKBONE_EPOCHS": 2,
    # loss weights / regularisation (C3)
    "FRAME_CE_WEIGHT": 0.25, "AUX_CE_WEIGHT": 0.3, "LABEL_SMOOTHING": 0.0, "LAMBDA_TC": 0.01, "LAMBDA_REC": 1.0, "DROP_PATH": 0.1, "DROPOUT": 0.1,
    "EMA_DECAY": 0.998, "EARLY_STOP_PATIENCE": 99,   # let the cosine schedule finish; best_auc.pth still keeps the best epoch
    # KD (F1)
    "LAMBDA_KD": 0.1, "TEMPERATURE": 3.0, "KD_START_EPOCH": 2, "KD_RAMP_EPOCHS": 2, "KD_WEIGHTING": "confidence",
    # augmentation curriculum (G1)
    "AUG_BLUR_PROB": 0.10, "AUG_JPEG_PROB": 0.30, "AUG_JPEG_QMIN": 50, "AUG_JPEG_QMAX": 98, "AUG_DOWNSCALE_PROB": 0.10, "AUG_DOWN_MIN": 0.6,
    "AUG_COLOR_PROB": 0.30, "AUG_NOISE_PROB": 0.05, "AUG_RAMP_EPOCHS": 3,
    # generative branch (E1)
    "GEN_FEED": "residual_cnn", "GEN_RES_DETACH": True, "GEN_PRETRAIN_CLEAN": True, "GEN_PRETRAIN_REAL_ONLY": False,
    # diagnostics (C2)
    "RUN_LOSS_DIAGNOSTIC": True,
    # frequency branch input: "fft" (global log-spectrum) | "dct" (8x8 block DCT, original)
    "FREQ_TYPE": "fft",
}
# =====================================================================================
#                     USER TEACHER HOOKS
# =====================================================================================
# ================= MODIFIED =================
def _teacher_state_dict(ck):
    sd = ck
    if isinstance(ck, dict):
        for k in ("teacher_state_dict", "model_state_dict", "state_dict", "model", "net", "ema"):
            if k in ck and isinstance(ck[k], dict): sd = ck[k]; break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

def _dpvit_layout(keys):
    """Top-level names of a dual-path ViT state dict: (rgb_branch_name, hf_branch_name, classifier_name). Raises if the layout is not 2 ViTs + 1 head."""
    keys = [k[7:] if k.startswith("module.") else k for k in keys]
    vit = sorted({k.split(".")[0] for k in keys if ".blocks." in k})
    if len(vit) != 2: raise RuntimeError(f"Expected 2 ViT branches in the teacher checkpoint, found {vit}.")
    rgb = [n for n in vit if "rgb" in n.lower()]
    if len(rgb) != 1: raise RuntimeError(f"Cannot tell which ViT branch is RGB among {vit} (expected exactly one name containing 'rgb').")
    hf = [n for n in vit if n != rgb[0]][0]
    head = sorted({k.split(".")[0] for k in keys} - set(vit))
    if len(head) != 1: raise RuntimeError(f"Expected exactly one classifier module besides {vit}, found {head}.")
    return rgb[0], hf, head[0]

class DPViTTeacher(nn.Module):
    """Dual-path ViT teacher (layout of 'DP ViT FF++.py'): ViT-B/16(RGB) + ViT-B/16(Laplacian) -> concat(1536) -> MLP -> ONE logit (fake > 0).
    Submodule names come from the checkpoint so state-dict keys match exactly."""
    def __init__(self, names):
        super().__init__(); self.names = tuple(names)
        for n in self.names[:2]: setattr(self, n, timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0))
        setattr(self, self.names[2], nn.Sequential(nn.Dropout(0.3), nn.Linear(1536, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 1)))
    def forward(self, x_rgb, x_hf):
        r, h, c = (getattr(self, n) for n in self.names)
        return c(torch.cat([r(x_rgb), h(x_hf)], 1))

def build_teacher_model(cfg):
    """DP-ViT teacher (RGB + Laplacian streams). Architecture/names are read from TEACHER_CHECKPOINT; loading in load_teacher() stays STRICT."""
    ck = load_ckpt(cfg["TEACHER_CHECKPOINT"])
    if isinstance(ck, nn.Module): return ck
    return DPViTTeacher(_dpvit_layout(_teacher_state_dict(ck).keys()))

# ================= MODIFIED =================
def teacher_forward(model, x01, cfg):
    """EDIT ME for a custom teacher. Called as teacher_forward(teacher.m, x01, cfg) by the TeacherNet wrapper.
    x01: float [N,3,S,S] in [0,1], already resized to TEACHER_INPUT_SIZE (no normalisation applied yet).
    Must return logits [N,2] (or [N,1]). DEFAULT (plain RGB teacher): ImageNet-style normalisation (TEACHER_MEAN/STD) then model(x).
    For a DP-ViT / multi-stream teacher: derive the extra inputs here from x01 (Laplacian / high-frequency map, extra streams ...), e.g.
        lap = my_laplacian(x01); return model(rgb_norm, lap)
    The wrapper takes care of: resize, fp32 cast, [N,1]->[N,2], TEACHER_FAKE_INDEX, torch.no_grad() and eval()."""
    if cfg.get("TEACHER_BGR", False): x01 = x01.flip(1)                                       # test: teacher trained on cv2/BGR images -> feed it BGR
    mean = torch.tensor(cfg["TEACHER_MEAN"], device=x01.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(cfg["TEACHER_STD"], device=x01.device, dtype=torch.float32).view(1, 3, 1, 1)
    with torch.autocast("cuda", enabled=False):
        xq = (x01.float() * 255.0).round()                                                    # back to uint8 values
        k = xq.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)   # cv2.Laplacian, ksize=1
        lap = F.conv2d(F.pad(xq, (1, 1, 1, 1), mode="reflect"), k, groups=3)                 # BORDER_REFLECT_101 == 'reflect'
        lap = lap.abs().round().clamp(0, 255)                                                 # cv2.convertScaleAbs
        x_rgb = (xq / 255.0 - mean) / std; x_hf = (lap / 255.0 - mean) / std                  # same ImageNet normalisation for both streams
    return model(x_rgb, x_hf)                                                                 # [N,1] logit; TeacherNet maps it to [0, logit] -> p_fake = sigmoid

# =====================================================================================
#                                   UTILITIES
# =====================================================================================
LOGFILE = {"path": None}
def elapsed_min(): return (time.time() - SESSION_START) / 60.0
def log(*a):
    s = f"[{elapsed_min():7.1f}m] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGFILE["path"]:
        with open(LOGFILE["path"], "a") as f: f.write(s + "\n")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def set_rng_state(st):
    try:
        random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"].cpu())
        if st.get("cuda") is not None and torch.cuda.is_available():
            for i, s in enumerate(st["cuda"][:torch.cuda.device_count()]): torch.cuda.set_rng_state(s.cpu(), i)
    except Exception as e: log("WARN: could not fully restore RNG state:", e)

def atomic_save(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def _dir_to_zip_ckpt(d):
    """Kaggle may expose a torch checkpoint as an UNPACKED folder (data.pkl, data/, byteorder, version, ...). Re-zip it (uncompressed) into a temp .pth that torch.load can read."""
    d = os.path.normpath(d); cache = f"/tmp/ckpt_zip_{hashlib.md5(d.encode()).hexdigest()[:10]}.pth"
    if os.path.exists(cache): return cache
    if not os.path.exists(os.path.join(d, "data.pkl")): raise FileNotFoundError(f"'{d}' is a folder but contains no data.pkl -> not an unpacked torch checkpoint. Contents: {sorted(os.listdir(d))[:20]}")
    tmp = cache + ".tmp"; log(f"Checkpoint '{d}' is an unpacked folder -> re-zipping to {cache}")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for dp, _, fn in os.walk(d):
            for f in sorted(fn): full = os.path.join(dp, f); z.write(full, "archive/" + os.path.relpath(full, d).replace(os.sep, "/"))
    os.replace(tmp, cache); return cache

def ckpt_size_mb(path):
    if os.path.isdir(path): return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(path) for f in fn) / 2 ** 20
    return os.path.getsize(path) / 2 ** 20

def load_ckpt(path):
    if os.path.isdir(path): path = _dir_to_zip_ckpt(path)
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")
def close_iter(it):
    """Explicitly shut down DataLoader workers and collect garbage, so a later fork never inherits a live iterator."""
    try: it._shutdown_workers()
    except AttributeError: pass
    gc.collect()

def gpu_mem():
    if not torch.cuda.is_available(): return "cpu"
    return " ".join(f"g{i}:{torch.cuda.max_memory_allocated(i)/2**30:.1f}G" for i in range(torch.cuda.device_count()))

def make_parallel(m, cfg):
    return nn.DataParallel(m) if (cfg["USE_DATAPARALLEL"] and torch.cuda.device_count() > 1) else m

def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m

def to_x01(x_u8, dev):   # uint8 [B,T,H,W,3] -> float [B,T,3,H,W] in [0,1]
    return x_u8.to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0)

def make_scaler(enabled):
    try: return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception: return torch.cuda.amp.GradScaler(enabled=enabled)

# ================= MODIFIED =================
def prerun_guard(name, fn, *a, **k):
    """PRE-RUN HARD STOP: any failure of a pre-run check aborts the run before training starts (the original exception is re-raised)."""
    try: return fn(*a, **k)
    except Exception as e:
        log(f"!! PRE-RUN HARD STOP: check '{name}' FAILED -> training NOT started. {type(e).__name__}: {e}"); raise

def model_tag(cfg): return f"T{int(cfg['USE_TEMPORAL'])}F{int(cfg['USE_FREQUENCY'])}A{int(cfg['USE_SPATIAL_ARTIFACT'])}KD{int(cfg['USE_KD'])}"

# =====================================================================================
#                          DATA: SCANNING, PARSING, GROUPING, SPLITTING
# =====================================================================================
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REAL_KEYS = ("real", "orginal", "original", "pristine", "youtube", "genuine", "authentic")
GROUP_MODES = ("auto", "video_dirs", "filename_prefix", "single_image")
GROUP_KEY = {"FF++": "FFPP_GROUPING_MODE", "CDF": "CELEBDF_GROUPING_MODE", "DFDC": "DFDC_GROUPING_MODE"}

def natural_key(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
def label_from_top(top): return 0 if any(k in top.lower() for k in REAL_KEYS) else 1

def resolve_root(root):
    root = os.path.normpath(root)
    for _ in range(4):
        ents = os.listdir(root)
        dirs = [e for e in ents if os.path.isdir(os.path.join(root, e))]
        imgs = [e for e in ents if e.lower().endswith(IMG_EXT)]
        if len(dirs) == 1 and not imgs: root = os.path.join(root, dirs[0])
        else: break
    return root

# ================= MODIFIED: cached file listing =================
CACHE_KEY = {"FF++": "ff", "CDF": "celeb", "DFDC": "dfdc", "DIFF": "diffusion"}
_SCAN_CACHE = {"loaded": False, "raw": None}

def _top_of(dp, root):
    rel = os.path.relpath(dp, root); return rel.split(os.sep)[0] if rel != "." else os.path.basename(root)

def _load_scan_cache(cfg):
    if _SCAN_CACHE["loaded"]: return _SCAN_CACHE["raw"]
    _SCAN_CACHE["loaded"] = True; p = cfg.get("SCAN_CACHE_PATH", "")
    if not cfg.get("USE_SCAN_CACHE") or not p or not os.path.exists(p):
        log("Scan cache not used (disabled or file not found) -> live os.walk scans."); return None
    try:
        with open(p, "rb") as f: raw = pickle.load(f)
        if not isinstance(raw, dict): raise TypeError(f"expected dict, got {type(raw).__name__}")
    except Exception as e:
        log(f"WARN: could not read scan cache '{p}' ({type(e).__name__}: {e}) -> live os.walk scans."); return None
    log(f"Scan cache loaded: {p} | " + ", ".join(f"{k}={len(v):,} frames" for k, v in raw.items()))
    _SCAN_CACHE["raw"] = raw; return raw

def cached_listing(ds_name, root, cfg):
    """Rebuild an os.walk-equivalent listing from the pickle (paths only; the stored video id is ignored on purpose).
    Returns (resolved_root, {dir: [image names]}, dirs_that_have_subdirs) or None -> caller does a live scan."""
    raw = _load_scan_cache(cfg); key = CACHE_KEY.get(ds_name)
    if raw is None or key not in raw: return None
    def reject(why): log(f"[{ds_name}] scan cache REJECTED ({why}) -> live os.walk scan."); return None
    try: entries = [(os.path.normpath(str(e[0])), int(e[1])) for e in raw[key]]
    except Exception as e: return reject(f"unexpected entry format: {e}")
    root0 = os.path.normpath(root); inside = sum(p.startswith(root0 + os.sep) for p, _ in entries)
    if not entries or inside != len(entries): return reject(f"{inside}/{len(entries)} cached paths lie under '{root0}'")
    probe = random.Random(0).sample(entries, min(cfg.get("SCAN_CACHE_VERIFY_N", 200), len(entries)))
    miss = sum(not os.path.exists(p) for p, _ in probe)
    if miss: return reject(f"{miss}/{len(probe)} sampled files do not exist on disk")
    listing = defaultdict(list)
    for p, _ in entries:
        if p.lower().endswith(IMG_EXT): listing[os.path.dirname(p)].append(os.path.basename(p))
    rroot = root0                                             # emulate resolve_root(): descend through single-subfolder chains
    for _ in range(4):
        subs = {os.path.relpath(d, rroot).split(os.sep)[0] for d in listing}
        if len(subs) == 1 and "." not in subs: rroot = os.path.join(rroot, subs.pop())
        else: break
    bad = sum(label_from_top(_top_of(os.path.dirname(p), rroot)) != lab for p, lab in entries if p.lower().endswith(IMG_EXT))
    if bad: return reject(f"{bad} cached labels disagree with the folder-name labelling")
    parents = set()
    for d in listing:
        a = d
        while a != rroot and a.startswith(rroot): a = os.path.dirname(a); parents.add(a)
    log(f"[{ds_name}] using scan cache: {sum(map(len, listing.values())):,} frames in {len(listing):,} directories (root {rroot})")
    return rroot, listing, parents

# ================= MODIFIED =================
def parse_name(fname, style):
    """file name -> (video prefix, frame number). '0001.jpg' -> ('', 1); 'vid_0001.jpg' -> ('vid', 1); 'frame0007.jpg' -> ('frame', 7)."""
    stem = os.path.splitext(fname)[0]; toks = stem.split("_")
    if style == "dfdc" and len(toks) >= 3 and toks[-1].isdigit() and toks[-2].isdigit():
        return "_".join(toks[:-2]) + "_f" + toks[-1], int(toks[-2])      # video_frame_face  -> one track per face
    m = re.match(r"^(.*?)[_\-. ]?(\d+)$", stem)
    if m: return m.group(1), int(m.group(2))
    return stem, 0

def _frame_sort_key(style): return lambda f: (parse_name(f, style)[1], natural_key(f))

def _auto_mode(ds_name, dp, files, has_sub, style):
    """Resolve the grouping of ONE directory that directly contains frames. Raises when uncertain (never guesses dangerously)."""
    groups = defaultdict(list)
    for f in files: groups[parse_name(f, style)[0]].append(f)
    npfx, nf = len(groups), len(files); sizes = [len(g) for g in groups.values()]; key = GROUP_KEY.get(ds_name, "<DATASET>_GROUPING_MODE")
    ex = sorted(files, key=natural_key)[:4]
    def fail(why):
        raise RuntimeError(f"[{ds_name}] AUTO grouping is UNCERTAIN for directory '{dp}': {why} ({nf} images, {npfx} distinct name prefixes, e.g. {ex}). "
                           f"Set CONFIG['{key}'] explicitly to 'video_dirs' (one directory = one video), 'filename_prefix' (videoID_frameID.ext) or 'single_image'.")
    if npfx == 1:
        if has_sub: fail("loose frames AND sub-directories in the same directory")
        return "video_dirs"                                   # leaf directory with one naming pattern -> ONE video (frame numbers are NOT video ids)
    if "" not in groups and float(np.median(sizes)) >= 2: return "filename_prefix"      # videoID_frameID.ext with several frames per id
    fail("file names look like independent images / mixed numbering")

def build_videos(ds_name, root, style="ffpp", group_mode="auto", max_per_folder=None, seed=0, cfg=None):
    if group_mode not in GROUP_MODES: raise ValueError(f"group_mode must be one of {GROUP_MODES}, got '{group_mode}'")
    cached = cached_listing(ds_name, root, cfg) if cfg is not None else None
    if cached is not None:
        root, listing, parents = cached
        walker = ((dp, (["_"] if dp in parents else []), listing[dp]) for dp in sorted(listing))   # same (dir, subdirs, files) shape as os.walk
    else:
        root = resolve_root(root); walker = os.walk(root)
    rng = random.Random(seed); videos = []; fkey = _frame_sort_key(style)
    for dp, dn, fn in walker:
        dn.sort(); files = [f for f in fn if f.lower().endswith(IMG_EXT)]
        if not files: continue
        rel = os.path.relpath(dp, root); top = rel.split(os.sep)[0] if rel != "." else os.path.basename(root)
        label = label_from_top(top); gen = top
        mode = _auto_mode(ds_name, dp, files, len(dn) > 0, style) if group_mode == "auto" else group_mode
        if mode == "single_image":
            files = sorted(files, key=natural_key)
            if max_per_folder and len(files) > max_per_folder: files = sorted(rng.sample(files, max_per_folder), key=natural_key)
            for f in files:
                videos.append(dict(video_id=f"{ds_name}/{rel}/{f}", vname=os.path.splitext(f)[0], label=label, generator=gen, dir=dp, files=[f], n=1, style=style, mode="single_image"))
        elif mode == "filename_prefix":
            groups = defaultdict(list)
            for f in files: groups[parse_name(f, style)[0]].append(f)
            for p, lst in groups.items():
                lst = sorted(lst, key=fkey); vname = p if p else (os.path.basename(rel) if rel != "." else top)
                videos.append(dict(video_id=f"{ds_name}/{rel}/{p}" if p else f"{ds_name}/{rel}", vname=vname, label=label, generator=gen, dir=dp, files=lst, n=len(lst), style=style, mode="filename_prefix"))
        else:                                                    # video_dirs: one directory == one video
            fl = sorted(files, key=fkey)
            videos.append(dict(video_id=f"{ds_name}/{rel}", vname=os.path.basename(rel) if rel != "." else top, label=label, generator=gen, dir=dp, files=fl, n=len(fl), style=style, mode="video_dirs"))
    videos.sort(key=lambda v: v["video_id"])
    if not videos: raise RuntimeError(f"No images found under {root}")
    return videos

def fast_val_subset(videos, n_real, n_fake_per_group, seed):
    """Deterministic stratified subset of the VALIDATION split (never the test split): n_real real videos + n_fake_per_group per manipulation folder."""
    by = defaultdict(list)
    for v in sorted(videos, key=lambda v: v["video_id"]): by[(v["label"], v["generator"])].append(v)
    rng = random.Random(seed); out = []
    for (lab, gen), vs in sorted(by.items()): out += rng.sample(vs, min(n_real if lab == 0 else n_fake_per_group, len(vs)))
    return sorted(out, key=lambda v: v["video_id"])

def describe(videos, name):
    n_fr = sum(v["n"] for v in videos); nr = sum(v["label"] == 0 for v in videos)
    gens = defaultdict(int)
    for v in videos: gens[v["generator"]] += 1
    log(f"[{name}] videos={len(videos)} frames={n_fr} real={nr} fake={len(videos)-nr} groups={dict(gens)}")

# ================= MODIFIED =================
def grouping_check(videos, name, cfg, mode_cfg="auto", n_show=3):
    """Prints grouping examples (dataset | video_id | directory | number_of_frames | first_frame | last_frame | label) and verifies plausibility."""
    log(f"--- video grouping check: {name} ---")
    modes = defaultdict(int)
    for v in videos: modes[v["mode"]] += 1
    print(f"grouping mode selected = {mode_cfg}   (resolved per directory: {dict(modes)})", flush=True)
    print("  dataset | video_id | directory | number_of_frames | first_frame | last_frame | label", flush=True)
    rng = random.Random(0)
    for lab in (0, 1):
        pool = [v for v in videos if v["label"] == lab]
        for v in rng.sample(pool, min(n_show, len(pool))):
            print(f"  {name} | {v['video_id']} | {v['dir']} | {v['n']} | {v['files'][0]} | {v['files'][-1]} | {'REAL(0)' if lab == 0 else 'FAKE(1)'}", flush=True)
    ids = [v["video_id"] for v in videos]
    if len(set(ids)) != len(ids): raise RuntimeError(f"{name}: duplicate video_ids -> grouping is unstable")
    image_mode = all(v["mode"] == "single_image" for v in videos)
    if not image_mode:
        bad_order = mixed = 0
        for v in videos:
            parsed = [parse_name(f, v["style"]) for f in v["files"]]; nums = [k for _, k in parsed]; uniform = len({p for p, _ in parsed}) == 1
            if not uniform: mixed += 1
            elif nums != sorted(nums): bad_order += 1
        if mixed: log(f"  WARNING: {mixed} videos contain several different name prefixes (mode '{mode_cfg}' was chosen explicitly).")
        if bad_order: raise RuntimeError(f"{name}: {bad_order} videos have non-chronological frame order.")
        ns = np.array([v["n"] for v in videos]); short = int((ns < cfg["NUM_FRAMES"]).sum())
        log(f"  videos={len(videos)} | frames/video: min={ns.min()} median={int(np.median(ns))} max={ns.max()} | videos with < {cfg['NUM_FRAMES']} frames: {short}")
        if float(np.median(ns)) < 2: raise RuntimeError(f"{name}: median video has <2 frames -> frames look like independent images; grouping is implausible. Set the dataset grouping mode explicitly.")
        subsampled = float(np.median(ns)) >= 2 and np.mean(ns == np.median(ns)) > 0.9     # >90% of videos share the same frame count -> pre-sampled frames per video, grouping is fine
        if short / len(videos) > 0.5 and not subsampled: raise RuntimeError(f"{name}: >50% of 'videos' have fewer than NUM_FRAMES frames -> frame grouping looks wrong.")
        if short: log(f"  WARNING: {short} videos shorter than NUM_FRAMES (frames will be repeated for those only).")
    log(f"  grouping OK: {len(videos)} unique video ids ({'independent images' if image_mode else 'frames chronologically ordered'}).")

# ================= MODIFIED =================
def ffpp_structure_diagnostic(videos, cfg, k=5):
    """DIAGNOSTIC ONLY: prints ~k representative REAL and FAKE FF++ examples (evenly spaced over the sorted video list) + the detected grouping. Saves nothing."""
    log("--- FF++ dataset structure diagnostic (file names only; no images are loaded or saved) ---")
    for lab, nm in ((0, "REAL"), (1, "FAKE")):
        pool = [v for v in videos if v["label"] == lab]; print(f"\n{nm}:  ({len(pool)} videos)", flush=True)
        if not pool: print("  (none found)", flush=True); continue
        for i in sorted(set(np.linspace(0, len(pool) - 1, min(k, len(pool))).round().astype(int).tolist())):
            v = pool[i]; print(f"  directory = {v['dir']}\n  video_id = {v['video_id']}\n  first_frame = {v['files'][0]}\n  last_frame = {v['files'][-1]}\n  n_frames = {v['n']}\n", flush=True)
    modes = defaultdict(int); gens = defaultdict(lambda: [None, 0])
    for v in videos: modes[v["mode"]] += 1; gens[v["generator"]][0] = v["label"]; gens[v["generator"]][1] += 1
    ns = np.array([v["n"] for v in videos])
    print("Detected FF++ grouping:", flush=True)
    print(f"  configured mode = {cfg['FFPP_GROUPING_MODE']} | resolved per directory = {dict(modes)}\n  videos = {len(videos)} | frames/video: min={ns.min()} median={int(np.median(ns))} max={ns.max()}", flush=True)
    print("  top-level folder -> (label, #videos): " + ", ".join(f"{g}->({'REAL' if l == 0 else 'FAKE'}, {n})" for g, (l, n) in sorted(gens.items())) + "\n", flush=True)

def ffpp_split_report(out, src):
    """Per-split video / real / fake / frame counts. For the OFFICIAL split the REAL (original) video counts MUST be exactly 720 / 140 / 140, otherwise STOP.
    Fake counts are NOT constrained (the extracted-frame dataset may contain only some manipulation types)."""
    exp = {"train": 720, "val": 140, "test": 140}; bad = []; print("\n=== FF++ split validation (" + ("OFFICIAL split" if src != "DEBUG_ONLY" else "DEBUG_ONLY split") + ") ===", flush=True)
    for sp in ("train", "val", "test"):
        vs = out[sp]; r = sum(v["label"] == 0 for v in vs); f = len(vs) - r
        print(f"{sp.upper()}:\n  total videos = {len(vs)}\n  real videos  = {r}" + (f"   (official = {exp[sp]})" if src != "DEBUG_ONLY" else "") + f"\n  fake videos  = {f}\n  total frames = {sum(v['n'] for v in vs)}", flush=True)
        if src != "DEBUG_ONLY" and r != exp[sp]: bad.append(f"{sp}: found {r} real videos, official = {exp[sp]}")
    if bad:
        real_dirs = sorted({v["generator"] for sp in out for v in out[sp] if v["label"] == 0})
        raise RuntimeError("FF++ official REAL-video counts are WRONG -> STOPPING before training. Found: " + "; ".join(bad) + f". Real-labelled top-level folders seen: {real_dirs}. "
                           "Likely causes: real frames missing from the dataset, real folder name not recognised as real, or wrong FFPP_GROUPING_MODE / video naming.")

# ---- official FF++ split ----
def load_official_splits(cfg, R):
    names = ("train", "val", "test"); data = {}; d = cfg["FFPP_SPLIT_DIR"]
    if d:
        if not os.path.isdir(d): raise FileNotFoundError(f"FFPP_SPLIT_DIR does not exist: {d}")
        for sp in names: data[sp] = json.load(open(os.path.join(d, f"{sp}.json")))
        src = d
    else:
        cache = os.path.join(R["configs"], "official_splits"); os.makedirs(cache, exist_ok=True); src = cfg["FFPP_SPLIT_URL_BASE"]
        for sp in names:
            fp = os.path.join(cache, f"{sp}.json")
            if not os.path.exists(fp):
                try:
                    raw = urllib.request.urlopen(f"{cfg['FFPP_SPLIT_URL_BASE']}/{sp}.json", timeout=30).read(); open(fp, "wb").write(raw)
                except Exception as e:
                    raise RuntimeError(f"Could not obtain the official FF++ {sp}.json ({e}). Turn Internet ON or upload train/val/test.json and set FFPP_SPLIT_DIR.")
            data[sp] = json.load(open(fp))
    exp_pairs = {"train": 360, "val": 70, "test": 70}; id2split = {}
    for sp in names:
        pairs = data[sp]
        if len(pairs) != exp_pairs[sp] or any(len(p) != 2 for p in pairs): raise RuntimeError(f"Official {sp}.json looks wrong: {len(pairs)} pairs (expected {exp_pairs[sp]}).")
        for p in pairs:
            for x in p:
                if int(x) in id2split: raise RuntimeError(f"Identity {x} appears in more than one official split entry")
                id2split[int(x)] = sp
    cnt = {sp: sum(1 for v in id2split.values() if v == sp) for sp in names}
    if cnt != {"train": 720, "val": 140, "test": 140}: raise RuntimeError(f"Official split sizes {cnt} != 720/140/140")
    log(f"Official FF++ split loaded from {src}: identities train/val/test = {cnt['train']}/{cnt['val']}/{cnt['test']}")
    return id2split, src

def debug_identity_split(videos, cfg):
    """DEBUG_ONLY: identity-graph split inferred from file names. NOT the official split; never use for the paper."""
    fr = cfg["SPLIT_FRACTIONS"]; seed = cfg["SEED"]; parent = {}
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for v in videos:
        for i in v["ids"]: parent.setdefault(i, i)
        if len(v["ids"]) == 2: union(*v["ids"])
    comps = defaultdict(list)
    for i in parent: comps[find(i)].append(i)
    clist = sorted(comps.values(), key=min); random.Random(seed).shuffle(clist)
    n = len(clist); n_val = max(1, round(fr[1] * n)); n_test = max(1, round(fr[2] * n)); id2split = {}
    for ci, comp in enumerate(clist):
        sp = "val" if ci < n_val else ("test" if ci < n_val + n_test else "train")
        for i in comp: id2split[i] = sp
    out = {"train": [], "val": [], "test": []}
    for v in videos:
        sps = {id2split.get(i) for i in v["ids"]}
        if len(sps) != 1 or None in sps: continue
        v["split"] = sps.pop(); out[v["split"]].append(v)
    return out

def _dhash(path, size=8):
    a = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

def dup_content_report(dups, n_probe=4, max_ids=200):
    """Content comparison of the two entries of a duplicated real id: min Hamming distance (of 64) between dHashes of n_probe evenly spaced frames of each entry."""
    cls_count = defaultdict(int); dists = []
    for k in sorted(dups)[:max_ids]:
        g = dups[k]
        if len(g) != 2: continue
        hs = [[_dhash(os.path.join(v["dir"], v["files"][j])) for j in np.linspace(0, v["n"] - 1, min(n_probe, v["n"])).round().astype(int)] for v in g]
        d = min(int((a != b).sum()) for a in hs[0] for b in hs[1]); dists.append(d)
        cls_count["identical/near-identical (<=4)" if d <= 4 else "near-duplicate (<=10)" if d <= 10 else "uncertain (<=20)" if d <= 20 else "different content (>20)"] += 1
    print(f"--- duplicate CONTENT check on {len(dists)} ids (dHash, 64 bits): median min-distance={np.median(dists):.1f} | {dict(cls_count)}", flush=True)
    print("    Only 'identical/near-identical' entries are provably the same frames. Otherwise the two entries are DIFFERENT extractions; the policy then chooses the one consistent with the fakes' pipeline, it is not claiming they are copies.", flush=True)

def shortcut_probe(videos, n=150, seed=0):
    """Non-forensic cues (image size, JPEG bytes/pixel, frames/video): AUC of each cue alone for real-vs-fake on TRAIN videos. AUC far from 0.5 = pipeline shortcut."""
    rng = random.Random(seed); rows = []
    for lab in (0, 1):
        pool = [v for v in videos if v["label"] == lab]
        for v in rng.sample(pool, min(n, len(pool))):
            p = os.path.join(v["dir"], v["files"][len(v["files"]) // 2])
            with Image.open(p) as im_: w, h = im_.size
            rows.append((lab, w, h, os.path.getsize(p) / (w * h), v["n"]))
    a = np.array(rows, dtype=float); y = a[:, 0]; log("--- shortcut probe (train videos; each cue alone) ---")
    for j, nm in ((1, "width"), (2, "height"), (3, "file bytes/pixel"), (4, "frames/video")):
        print(f"  {nm:17s} real mean={a[y == 0, j].mean():9.3f} | fake mean={a[y == 1, j].mean():9.3f} | AUC of this cue alone = {roc_auc_score(y, a[:, j]):.3f}", flush=True)

def dedupe_real(videos, cfg):
    """Real (original) videos are identified by ONE numeric id. If the frame dataset holds several entries per id (e.g. two copies of the extracted
    frames), print where they live and keep exactly one per id according to REAL_DUPLICATE_POLICY. Fake videos are never touched."""
    pol = cfg["REAL_DUPLICATE_POLICY"]
    if pol not in ("keep_longest", "keep_first", "error", "match_fake_pipeline"): raise ValueError("REAL_DUPLICATE_POLICY must be keep_longest | keep_first | error | match_fake_pipeline")
    groups = defaultdict(list)
    for v in videos:
        if v["label"] == 0 and v["ids"]: groups[v["ids"][0]].append(v)
    dups = {k: g for k, g in groups.items() if len(g) > 1}
    if not dups: return videos
    sizes = defaultdict(int)
    for g in dups.values(): sizes[len(g)] += 1
    parents = defaultdict(int)
    for g in dups.values():
        for v in g: parents[os.path.dirname(v["dir"])] += 1
    print(f"\n--- DUPLICATE REAL VIDEO IDS: {len(dups)} ids have several entries (entries-per-id histogram: {dict(sizes)}) ---", flush=True)
    print("  parent directories holding these entries (path -> #entries):", flush=True)
    for p, c in sorted(parents.items()): print(f"      {p} -> {c}", flush=True)
    for k in sorted(dups)[:5]:
        print(f"  id {k}:", flush=True)
        for v in dups[k]: print(f"      dir={v['dir']} | mode={v['mode']} | n_frames={v['n']} | first={v['files'][0]} | last={v['files'][-1]}", flush=True)
    same_n = sum(1 for g in dups.values() if len({v["n"] for v in g}) == 1)
    print(f"  ids whose copies have identical frame counts: {same_n}/{len(dups)}  (high = true copies; low = probably different content -> check before trusting 'keep_*')\n", flush=True)
    if pol == "error": raise RuntimeError("Duplicate real video ids found (see DUPLICATE REAL VIDEO IDS above). Set REAL_DUPLICATE_POLICY to 'keep_longest' or 'keep_first' after inspecting them.")
    if cfg.get("DUP_CONTENT_CHECK", True): dup_content_report(dups)
    fake_n = defaultdict(list)
    for v in videos:
        if v["label"] == 1 and v["ids"]: fake_n[v["ids"][0]].append(v["n"])      # fake 'T_S' follows target video T -> same extraction as real id T
    drop = set(); matched = fallback = 0
    for k, g in dups.items():
        if pol == "match_fake_pipeline":
            score = lambda v: sum(1 for n in fake_n.get(k, []) if n == v["n"])
            best = max(score(v) for v in g); cand = [v for v in g if score(v) == best]
            if best > 0 and len(cand) == 1: keep = cand[0]; matched += 1
            else: keep = max(g, key=lambda v: (v["n"], v["video_id"])); fallback += 1
        else:
            keep = max(g, key=lambda v: (v["n"], v["video_id"])) if pol == "keep_longest" else min(g, key=lambda v: v["video_id"])
        drop.update(id(v) for v in g if v is not keep)
    if pol == "match_fake_pipeline": log(f"match_fake_pipeline: {matched} duplicated real ids resolved by frame-count agreement with their fake videos, {fallback} fell back to keep_longest")
    log(f"REAL_DUPLICATE_POLICY='{pol}': kept one entry per real id, discarded {len(drop)} duplicate real entries.")
    return [v for v in videos if id(v) not in drop]

def ffpp_split(videos, cfg, R):
    ids_of = lambda s: [int(x) for x in re.findall(r"\d+", s)][:2]
    for v in videos: v["ids"] = ids_of(v["vname"])[:1] if v["label"] == 0 else ids_of(v["vname"])   # real = ONE source id; fake = (target, source) pair
    videos = dedupe_real(videos, cfg)
    try: id2split, src = load_official_splits(cfg, R)
    except Exception as e:
        if not cfg["ALLOW_DEBUG_SPLIT"]: raise
        log(f"WARNING: DEBUG FALLBACK — official split unavailable ({e}). Using DEBUG_ONLY identity-inferred split. NOT valid for paper results.")
        out = debug_identity_split(videos, cfg); src = "DEBUG_ONLY"
    else:
        out = {"train": [], "val": [], "test": []}; dropped = unknown = 0; drop_list = []
        for v in videos:
            if not v["ids"]: unknown += 1; continue
            sps = {id2split.get(i) for i in v["ids"]}
            if len(sps) != 1 or None in sps: dropped += 1; drop_list.append((v, sps)); continue
            v["split"] = sps.pop(); out[v["split"]].append(v)
        if unknown: raise RuntimeError(f"{unknown} FF++ videos have no numeric id in their name; cannot map them to the official split.")
        if dropped:
            log(f"WARNING: {dropped} videos dropped (ids not in a single official split)")
            by_gen = defaultdict(list)
            for v, sps in drop_list: by_gen[(v["generator"], v["label"])].append((v, sps))
            print("\n--- DROPPED-VIDEO DIAGNOSTIC (folder | label | #dropped | examples) ---", flush=True)
            for (g_, lab_), lst in sorted(by_gen.items()):
                tot_ = sum(1 for v in videos if v["generator"] == g_)
                print(f"  {g_} | {'REAL' if lab_ == 0 else 'FAKE'} | {len(lst)} of {tot_} videos dropped", flush=True)
                for v, sps in lst[:6]:
                    print(f"      vname='{v['vname']}' ids={v['ids']} splits={sorted(map(str, sps))} n_frames={v['n']} dir={v['dir']}", flush=True)
            ok_real = [v for v in videos if v["label"] == 0 and "split" in v]
            print(f"  REAL videos mapped OK: {len(ok_real)} (official = 1000)\n", flush=True)
        if dropped > 0.01 * len(videos): raise RuntimeError("Too many videos do not map to the official split -> dataset naming differs from FF++. See the DROPPED-VIDEO DIAGNOSTIC printed above.")
    ffpp_split_report(out, src)        # exact official REAL-video counts (720/140/140) or STOP; prints videos/real/fake/frames per split
    vids = {s: {v["video_id"] for v in out[s]} for s in out}; idset = {s: {i for v in out[s] for i in v["ids"]} for s in out}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (vids[a] & vids[b]), f"VIDEO LEAKAGE {a}/{b}"
        assert not (idset[a] & idset[b]), f"IDENTITY LEAKAGE {a}/{b}"
    log(f"LEAK CHECK PASSED ({'official split' if src != 'DEBUG_ONLY' else 'DEBUG_ONLY split'}): splits disjoint at video AND identity level.")
    for s in out: describe(out[s], f"FF++ {s}")
    sig = hashlib.md5("|".join(sorted(vids["val"]) + ["#"] + sorted(vids["test"])).encode()).hexdigest()[:12]
    return out, sig

# ---- temporal sampling ----
def eval_window_starts(n, W, hop, max_clips):
    if n <= W: return [0], n
    st = list(range(0, n - W + 1, hop))
    if st[-1] + W < n: st.append(n - W)
    if len(st) > max_clips:
        st = [st[len(st) // 2]] if max_clips == 1 else [st[i] for i in sorted(set(np.linspace(0, len(st) - 1, max_clips).round().astype(int).tolist()))]
    return st, W

def eval_clip_plan(n, T, stride, max_clips):
    """Deterministic evaluation clips covering the WHOLE video with the SAME frame stride as training.
    Returns [(start, window)]. window == span -> frames start, start+stride, ... (no unused frames inside the window).
    K = min(max_clips, ceil(n/span)) clips, starts evenly spaced from frame 0 to the last possible start (first and last frames are reachable)."""
    span = (T - 1) * stride + 1
    if n <= span: return [(0, n)]                              # short video: clip_indices() spreads T frames uniformly over the entire video
    k = max(1, min(int(max_clips), int(math.ceil(n / span))))
    if k == 1: return [((n - span) // 2, span)]                # single clip -> centred
    starts = sorted(set(int(round(s)) for s in np.linspace(0, n - span, k)))
    return [(s, span) for s in starts]

def clip_indices(n, T, w0, W, stride, off):
    if n < T: return [int(round(i)) for i in np.linspace(0, n - 1, T)]     # unavoidable repetition (video shorter than T)
    span = (T - 1) * stride + 1
    if span <= W: return [w0 + off + stride * i for i in range(T)]         # ordered, strided, inside the window
    return [w0 + int(round(i)) for i in np.linspace(0, W - 1, T)]          # window too small for the stride -> uniform in window

# ================= MODIFIED =================
class ClipDataset(Dataset):
    """Train: one random temporal window per item. Eval: several windows per video (see eval_window_starts).
    TRAINING IS DETERMINISTIC: every random decision (window, offset, crop, flip, blur, JPEG) comes from a local RNG seeded by
    (SEED, current_epoch, draw position, item index) -> no worker-local RNG state; identical after a mid-epoch resume."""
    def __init__(self, videos, cfg, train, max_clips=1):
        self.v, self.cfg, self.train = videos, cfg, train
        self.T, self.S = cfg["NUM_FRAMES"], cfg["IMAGE_SIZE"]; self.items = []; self.current_epoch = 0
        self.stride = 1 if cfg["TEMPORAL_SAMPLING"] == "contiguous" else cfg["TEMPORAL_STRIDE"]
        for vi, v in enumerate(videos):
            if train: self.items.append((vi, None, None))
            elif cfg.get("EVAL_CLIP_MODE", "cover") == "cover":
                for s, W in eval_clip_plan(v["n"], self.T, self.stride, max_clips): self.items.append((vi, s, W))
            else:                                                    # legacy mode (kept as an ablation)
                starts, W = eval_window_starts(v["n"], cfg["EVAL_WINDOW"], cfg["EVAL_WINDOW_STRIDE"], max_clips)
                for s in starts: self.items.append((vi, s, W))
    def set_epoch(self, epoch): self.current_epoch = int(epoch)
    def clip_rng(self, pos, item):
        return np.random.default_rng([int(self.cfg["SEED"]), int(self.current_epoch), int(pos), int(item)])
    def __len__(self): return len(self.items)
    def indices(self, vi, w0, W, rng): return self.indices_ex(vi, w0, W, rng)[0]
    def indices_ex(self, vi, w0, W, rng):
        """Same sampling as before (identical RNG draw order); additionally returns the chosen window {w0, We, off} for the temporal sanity check."""
        n = self.v[vi]["n"]; T = self.T; span = (T - 1) * self.stride + 1
        if self.train:
            We = min(self.cfg["TRAIN_WINDOW"], n); w0 = int(rng.integers(0, n - We + 1))
            off = int(rng.integers(0, We - span + 1)) if (n >= T and span <= We) else 0
        else:
            We = W; off = (We - span) // 2 if (n >= T and span <= We) else 0
        return [min(max(i, 0), n - 1) for i in clip_indices(n, T, w0, We, self.stride, off)], dict(w0=int(w0), We=int(We), off=int(off))
    def meta(self, i):
        vi, w0, W = self.items[i]; v = self.v[vi]; idx = self.indices(vi, w0, W, None)
        return v, [v["files"][j] for j in idx]
    def _make_aug(self, rng):
        """ONE set of degradation parameters per CLIP (so the temporal branch never sees augmentation flicker). Deterministic given the clip RNG."""
        c = self.cfg; ramp = c.get("AUG_RAMP_EPOCHS", 0); r = 1.0 if ramp <= 0 else min(1.0, self.current_epoch / ramp)
        return dict(blur=float(rng.uniform(0.3, 1.5)) if rng.random() < c["AUG_BLUR_PROB"] * r else None,
                    jpeg=int(rng.integers(c.get("AUG_JPEG_QMIN", 50), c.get("AUG_JPEG_QMAX", 98))) if rng.random() < c["AUG_JPEG_PROB"] * r else None,
                    down=float(rng.uniform(c.get("AUG_DOWN_MIN", 0.6), 0.85)) if rng.random() < c["AUG_DOWNSCALE_PROB"] * r else None,
                    color=tuple(float(v) for v in rng.uniform(0.75, 1.25, 3)) if rng.random() < c["AUG_COLOR_PROB"] * r else None,
                    noise=float(rng.uniform(2.0, 8.0)) if rng.random() < c["AUG_NOISE_PROB"] * r else None,
                    noise_seed=int(rng.integers(0, 2 ** 31 - 1)))
    def _make_sbi(self, rng):
        return dict(cx=float(rng.uniform(0.45, 0.55)), cy=float(rng.uniform(0.47, 0.60)), rx=float(rng.uniform(0.26, 0.38)), ry=float(rng.uniform(0.32, 0.45)),
                    rot=float(rng.uniform(-0.3, 0.3)), a1=float(rng.uniform(0, 0.10)), p1=float(rng.uniform(0, 6.28)), a2=float(rng.uniform(0, 0.06)), p2=float(rng.uniform(0, 6.28)),
                    msig=float(rng.uniform(0.01, 0.05)), alpha=float(rng.uniform(0.6, 1.0)), scale=float(rng.uniform(0.93, 1.07)),
                    tx=float(rng.uniform(-0.03, 0.03)), ty=float(rng.uniform(-0.03, 0.03)), color=tuple(float(v) for v in rng.uniform(0.8, 1.2, 3)), sblur=float(rng.uniform(0, 1.2)))
    def _self_blend(self, im, q):
        S = self.S; yy, xx = np.mgrid[0:S, 0:S].astype(np.float32); x = xx / S - q["cx"]; y = yy / S - q["cy"]
        c, s = math.cos(q["rot"]), math.sin(q["rot"]); xr = c * x + s * y; yr = -s * x + c * y; th = np.arctan2(yr, xr)
        rad = 1.0 + q["a1"] * np.sin(th + q["p1"]) + q["a2"] * np.sin(2 * th + q["p2"])
        m = (((xr / q["rx"]) ** 2 + (yr / q["ry"]) ** 2) <= rad ** 2).astype(np.uint8) * 255
        mk = np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(max(1.0, q["msig"] * S))), dtype=np.float32)[..., None] / 255.0 * q["alpha"]
        k = 1.0 / q["scale"]; src = im.transform((S, S), Image.AFFINE, (k, 0, S / 2 * (1 - k) + q["tx"] * S, 0, k, S / 2 * (1 - k) + q["ty"] * S), resample=Image.BILINEAR)
        b_, c_, s_ = q["color"]; src = ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(src).enhance(b_)).enhance(c_)).enhance(s_)
        if q["sblur"] > 0.3: src = src.filter(ImageFilter.GaussianBlur(q["sblur"]))
        a = np.asarray(im, np.float32); b = np.asarray(src, np.float32)
        return Image.fromarray(np.clip(a * (1 - mk) + b * mk, 0, 255).astype(np.uint8))
    def _prep(self, im, p, rng, k=0):
        aug = None
        if p is not None:
            s, ox, oy, flip, aug = p; W, H = im.size; cw, ch = max(8, int(W * s)), max(8, int(H * s))
            x0, y0 = int((W - cw) * ox), int((H - ch) * oy); im = im.crop((x0, y0, x0 + cw, y0 + ch))
            if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.resize((self.S, self.S), Image.BILINEAR)
        if self.train and aug is not None:
            if aug.get("sbi") is not None: im = self._self_blend(im, aug["sbi"])
            if aug["down"]: d = max(32, int(self.S * aug["down"])); im = im.resize((d, d), Image.BILINEAR).resize((self.S, self.S), Image.BILINEAR)
            if aug["blur"]: im = im.filter(ImageFilter.GaussianBlur(aug["blur"]))
            if aug["color"]:
                b_, c_, s_ = aug["color"]; im = ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(b_)).enhance(c_)).enhance(s_)
            if aug["noise"]:
                g_ = np.random.default_rng(aug["noise_seed"] + 7919 * k)
                arr = np.asarray(im, dtype=np.float32) + g_.normal(0.0, aug["noise"], (self.S, self.S, 3)).astype(np.float32)
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            if aug["jpeg"]:
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=aug["jpeg"]); buf.seek(0); im = Image.open(buf).convert("RGB")
        return np.asarray(im, dtype=np.uint8)
    def __getitem__(self, key):
        # key = item index (eval / preflight)  OR  (draw_position, item_index) from the epoch sampler (training)
        pos, i = (int(key[0]), int(key[1])) if isinstance(key, (tuple, list)) else (int(key), int(key))
        vi, w0, W = self.items[i]; v = self.v[vi]; rng = None; p = None; fk = False
        if self.train:
            rng = self.clip_rng(pos, i); p = (rng.uniform(0.8, 1.0), rng.random(), rng.random(), rng.random() < 0.5, self._make_aug(rng))
            if v["label"] == 0 and self.cfg.get("SBI_PROB", 0.0) > 0 and rng.random() < self.cfg["SBI_PROB"]: p[4]["sbi"] = self._make_sbi(rng); fk = True
        idx = self.indices(vi, w0, W, rng); frames = []
        for j in idx:
            try: im = Image.open(os.path.join(v["dir"], v["files"][j])).convert("RGB")
            except Exception: im = Image.new("RGB", (self.S, self.S)) if not frames else None
            frames.append(self._prep(im, p, rng, len(frames)) if im is not None else frames[-1])
        return torch.from_numpy(np.stack(frames)), torch.tensor(1 if fk else v["label"], dtype=torch.long), i

# =====================================================================================
#                                       MODEL
# =====================================================================================
def dct_matrix(N=8):
    k = torch.arange(N).float().unsqueeze(1); n = torch.arange(N).float().unsqueeze(0)
    D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N)) * math.sqrt(2.0 / N); D[0] /= math.sqrt(2.0); return D

class BlockDCT(nn.Module):
    """8x8 block 2-D DCT-II of luminance: [N,3,H,W] in [0,1] -> [N,64,H/8,W/8], signed-log compressed (fp32)."""
    def __init__(self, B=8):
        super().__init__(); self.B = B; self.register_buffer("K", torch.kron(dct_matrix(B), dct_matrix(B)))
    def forward(self, x01):
        with torch.autocast("cuda", enabled=False):
            x = x01.float(); y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
            y = (y - 0.5) * 255.0; N, _, H, W = y.shape
            p = F.unfold(y, self.B, stride=self.B); c = torch.matmul(self.K.float(), p)
            c = torch.sign(c) * torch.log1p(c.abs())
            return torch.nan_to_num(c).view(N, self.B * self.B, H // self.B, W // self.B)

class LogFFT(nn.Module):
    """Global 2-D FFT log-magnitude of Hann-windowed luminance: [N,3,H,W] in [0,1] -> [N,1,H,W] (fp32). Robust to crop/resize, unlike 8x8 block DCT."""
    def forward(self, x01):
        with torch.autocast("cuda", enabled=False):
            x = x01.float(); y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
            y = y - y.mean((2, 3), keepdim=True)
            win = torch.hann_window(y.shape[-2], device=y.device).view(1, 1, -1, 1) * torch.hann_window(y.shape[-1], device=y.device).view(1, 1, 1, -1)
            mag = torch.fft.fftshift(torch.fft.fft2(y * win), dim=(-2, -1)).abs()
            return torch.nan_to_num(torch.log1p(mag))

NORM = {"type": "gn"}
def gnorm(c, max_groups=8):
    if NORM["type"] == "bn": return nn.BatchNorm2d(c)
    g = max_groups
    while c % g: g -= 1
    return nn.GroupNorm(g, c)
def in_norm(c): return nn.BatchNorm2d(c) if NORM["type"] == "bn" else nn.GroupNorm(1, c)
    
class AttnPool(nn.Module):
    """Spatial attention pooling: feature map [N,C,h,w] -> D-dim embedding (keeps spatial selectivity, unlike mean/max pooling)."""
    def __init__(self, c, D):
        super().__init__(); self.score = nn.Conv2d(c, 1, 1); self.proj = nn.Sequential(nn.Linear(c, D), nn.GELU(), nn.LayerNorm(D))
    def forward(self, x):
        w = torch.softmax(self.score(x).flatten(2).float(), -1)           # [N,1,hw]
        f = (x.flatten(2).float() * w).sum(-1)                            # [N,C]
        return self.proj(f.to(x.dtype))

class FreqEncoder(nn.Module):
    def __init__(self, D, in_ch=64, widths=(64, 128, 192), strides=(1, 2, 1)):
        super().__init__(); self.bn_in = in_norm(in_ch); L = []; c = in_ch
        for w, s in zip(widths, strides): L += [nn.Conv2d(c, w, 3, stride=s, padding=1, bias=False), gnorm(w), nn.GELU()]; c = w
        self.net = nn.Sequential(*L); self.pool = AttnPool(c, D)              # map is 14x14 before pooling
    def forward(self, f): return self.pool(self.net(self.bn_in(f)))

class SpatialArtifactEncoder(nn.Module):
    """Fixed high-pass residual (Laplacian + SRM-style KV + 2nd-order) on RGB -> bounded residual -> small CNN -> attention pooling -> D.
    Operates on the RESIDUAL representation (not on RGB) to expose blending boundaries / local texture inconsistencies."""
    def __init__(self, D, widths=(32, 64, 96)):
        super().__init__()
        def pad5(k): k = torch.tensor(k, dtype=torch.float32); p = (5 - k.shape[0]) // 2; return F.pad(k, (p, p, p, p))
        lap = pad5([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kv = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
        so = pad5([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4.0
        K = torch.stack([lap, kv, so]).unsqueeze(1)                     # [3,1,5,5]
        self.register_buffer("hp", K.repeat(3, 1, 1, 1))                # [9,1,5,5] depthwise (groups=3): 3 filters per RGB channel
        self.bn_in = in_norm(9); L = []; c = 9
        for w in widths: L += [nn.Conv2d(c, w, 3, stride=2, padding=1, bias=False), gnorm(w), nn.GELU()]; c = w
        self.net = nn.Sequential(*L); self.pool = AttnPool(c, D)        # 224 -> 28x28 map
    def residual(self, x01):
        with torch.autocast("cuda", enabled=False):
            r = F.conv2d(x01.float(), self.hp, padding=2, groups=3)
            return torch.tanh(4.0 * torch.nan_to_num(r))                # bounded residual
    def forward(self, x01): return self.pool(self.net(self.bn_in(self.residual(x01))))

class ResidualEncoder(nn.Module):
    """Small CNN on the reconstruction residual (x - rec), tanh-bounded -> attention pooling -> D. Reads the forensic signal without a backbone pass."""
    def __init__(self, D, widths=(32, 64, 96)):
        super().__init__(); self.bn_in = in_norm(3); L = []; c = 3
        for w in widths: L += [nn.Conv2d(c, w, 3, stride=2, padding=1, bias=False), gnorm(w), nn.GELU()]; c = w
        self.net = nn.Sequential(*L); self.pool = AttnPool(c, D)
    def forward(self, r): return self.pool(self.net(self.bn_in(r)))

class AEBranch(nn.Module):
    """GenConViT-style generative branch: conv encoder -> (VAE latent) -> deconv decoder. 224 -> 7x7x256 -> 224."""
    def __init__(self, vae):
        super().__init__(); self.vae = vae; ch = [3, 16, 32, 64, 128, 256]; E = []
        for i in range(5): E += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), gnorm(ch[i + 1]), nn.ReLU(True)]
        self.enc = nn.Sequential(*E)
        if vae: self.mu = nn.Conv2d(256, 128, 1); self.lv = nn.Conv2d(256, 128, 1); self.fromz = nn.Conv2d(128, 256, 1)
        dch = [256, 128, 64, 32, 16]; Dd = []
        for i in range(4): Dd += [nn.ConvTranspose2d(dch[i], dch[i + 1], 4, 2, 1), gnorm(dch[i + 1]), nn.ReLU(True)]
        Dd += [nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid()]; self.dec = nn.Sequential(*Dd)
    def forward(self, x01):
        h = self.enc(x01)
        if self.vae:
            mu, lv = self.mu(h).float(), self.lv(h).float().clamp(-10, 10)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp(), dim=(1, 2, 3)); h = self.fromz(z.to(h.dtype))
        else: kl = torch.zeros(x01.shape[0], device=x01.device)
        return self.dec(h), kl

class Backbones(nn.Module):
    def __init__(self, names, pretrained, grad_ckpt, allow_fallback, drop_path=0.0):
        super().__init__(); nets = []
        for n in names:
            try: m = timm.create_model(n, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
            except Exception as e:
                if pretrained and not allow_fallback:
                    raise RuntimeError(f"Could not load PRETRAINED '{n}' ({type(e).__name__}: {e}). Turn Internet ON, or set ALLOW_DEBUG_FALLBACK=True for a DEBUG run with random init.")
                log(f"WARNING: DEBUG FALLBACK — '{n}' uses RANDOM INIT (ALLOW_DEBUG_FALLBACK=True). Do not use for paper results.")
                m = timm.create_model(n, pretrained=False, num_classes=0, drop_path_rate=drop_path)
            if grad_ckpt and hasattr(m, "set_grad_checkpointing"): m.set_grad_checkpointing(True)
            nets.append(m)
        self.nets = nn.ModuleList(nets); self.out_dim = sum(m.num_features for m in nets)
    def forward(self, x): return torch.cat([m(x) for m in self.nets], 1)

# ================= MODIFIED =================
class GenConViTFeat(nn.Module):
    """LIGHTWEIGHT GenConViT-BASED backbone. NOT an exact reproduction of the original 695M-parameter GenConViT implementation.
    Preserves the defining ingredients: Autoencoder (ED) and/or Variational Autoencoder (VAE) generative branches, whose
    reconstructions together with the original image pass through ConvNeXt-Tiny + Swin-Tiny feature extractors (pretrained,
    weights shared across passes). Features (incl. generative reconstruction information) are concatenated and projected to D.
    GENERATIVE-INPUT WARM-UP: the AE/VAE branches are learned jointly (randomly initialised), so during the first GENERATIVE_INPUT_WARMUP_EPOCHS epochs the backbone receives
    (1-gen_alpha)*original + gen_alpha*reconstruction for each generative branch (gen_alpha: 0 -> 1). The original-RGB pass is NEVER mixed. gen_alpha=1.0 (default, used for
    validation / inference) feeds the pure reconstruction. L_REC / L_KL always use the RAW reconstruction: MSE(reconstruction, original)."""
    def __init__(self, cfg, D):
        super().__init__(); v = cfg["GENCONVIT_VARIANT"]
        assert v in ("none", "ed", "vae", "both") and cfg["IMAGE_SIZE"] % 32 == 0
        if any("swin" in b for b in cfg["BACKBONES"]): assert cfg["IMAGE_SIZE"] == 224, "Swin-T backbone expects 224x224"
        self.backbones = Backbones(cfg["BACKBONES"], cfg["PRETRAINED"], cfg["GRAD_CHECKPOINT"], cfg["ALLOW_DEBUG_FALLBACK"], cfg.get("DROP_PATH", 0.0))
        self.branches = nn.ModuleList(([AEBranch(False)] if v in ("ed", "both") else []) + ([AEBranch(True)] if v in ("vae", "both") else []))
        self.feed = cfg.get("GEN_FEED", "residual_cnn"); assert self.feed in ("recon", "residual_cnn"); self.res_detach = cfg.get("GEN_RES_DETACH", True)
        if self.feed == "residual_cnn":
            self.resenc = nn.ModuleList([ResidualEncoder(D) for _ in self.branches]); fin = self.backbones.out_dim + len(self.branches) * D
        else: fin = self.backbones.out_dim * (1 + len(self.branches))
        self.proj = nn.Sequential(nn.LayerNorm(fin), nn.Linear(fin, D), nn.GELU(), nn.Dropout(cfg["DROPOUT"]), nn.Linear(D, D), nn.LayerNorm(D))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x01, gen_alpha=1.0):
        imgs = [(x01 - self.mean) / self.std]; rec_l = []; kl_l = []; res_f = []
        for bi, br in enumerate(self.branches):
            rec, kl = br(x01); rec_l.append(((rec.float() - x01.float()) ** 2).mean((1, 2, 3))); kl_l.append(kl.float())      # L_REC uses the RAW reconstruction
            if self.feed == "residual_cnn":                                                                                   # forensic residual -> small CNN (no extra backbone pass)
                r = x01.float() - rec.float()
                if self.res_detach: r = r.detach()                                                                            # AE stays a pure reconstructor (trained by L_REC only)
                res_f.append(self.resenc[bi](torch.tanh(8.0 * r)))
            else:
                bb_in = rec if gen_alpha >= 1.0 else (1.0 - gen_alpha) * x01 + gen_alpha * rec                                # generative-input warm-up (recon feed only)
                imgs.append((bb_in - self.mean) / self.std)
        n = x01.shape[0]; f = self.backbones(torch.cat(imgs, 0))
        f = f.view(len(imgs), n, -1).permute(1, 0, 2).reshape(n, -1); z = torch.zeros(n, device=x01.device)
        if res_f: f = torch.cat([f] + [t.to(f.dtype) for t in res_f], 1)
        return self.proj(f), (sum(rec_l) if rec_l else z), (sum(kl_l) if kl_l else z)

class TemporalEncoder(nn.Module):
    def __init__(self, D, layers, heads, drop, max_len=64):
        super().__init__(); self.cls = nn.Parameter(torch.zeros(1, 1, D)); self.pos = nn.Parameter(torch.zeros(1, max_len + 1, D))
        nn.init.trunc_normal_(self.cls, std=0.02); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(D, heads, D * 2, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(D)
    def forward(self, z):
        B, T, _ = z.shape; tok = torch.cat([self.cls.expand(B, -1, -1).to(z.dtype), z], 1) + self.pos[:, :T + 1].to(z.dtype)
        o = self.norm(self.enc(tok)); return o[:, 0], o[:, 1:]
    @torch.no_grad()
    def first_layer_attention(self, z):
        B, T, _ = z.shape; tok = torch.cat([self.cls.expand(B, -1, -1).to(z.dtype), z], 1) + self.pos[:, :T + 1].to(z.dtype)
        l = self.enc.layers[0]; x = l.norm1(tok); return l.self_attn(x, x, x, need_weights=True, average_attn_weights=True)[1]

GATE_IDX = {"spatial": 0, "artifact": 1, "frequency": 2}

class StudentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__(); D = cfg["EMBED_DIM"]; self.amp = cfg["USE_AMP"]; self.D = D
        self.use_a, self.use_f, self.use_t = cfg["USE_SPATIAL_ARTIFACT"], cfg["USE_FREQUENCY"], cfg["USE_TEMPORAL"]
        self.gen = GenConViTFeat(cfg, D)
        self.streams = ["spatial"] + (["artifact"] if self.use_a else []) + (["frequency"] if self.use_f else [])
        if self.use_a: self.aenc = SpatialArtifactEncoder(D)
        if self.use_f:
            if cfg.get("FREQ_TYPE", "dct") == "fft": self.dct = LogFFT(); self.fenc = FreqEncoder(D, in_ch=1, widths=(32, 64, 96), strides=(2, 2, 2))   # 224 -> 28x28 map
            else: self.dct = BlockDCT(8); self.fenc = FreqEncoder(D)
        # 3-way gated fusion (softmax over active streams, per channel)
        if len(self.streams) > 1: self.gate = nn.Linear(len(self.streams) * D, len(self.streams) * D)
        if self.use_t: self.temporal = TemporalEncoder(D, cfg["TEMP_LAYERS"], cfg["TEMP_HEADS"], cfg["DROPOUT"])
        head = lambda: nn.Sequential(nn.LayerNorm(D), nn.Dropout(cfg["DROPOUT"]), nn.Linear(D, 2))
        self.frame_head, self.video_head = head(), head()
        self.aux_heads = nn.ModuleDict({nm: nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2)) for nm in self.streams})
    def forward(self, x01, extras=False, debug=False, gen_alpha=1.0):
        ex = {}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            B, T = x01.shape[:2]; x = x01.flatten(0, 1); N = B * T; dev = x.device
            s, rec, kl = self.gen(x, gen_alpha); feats = [s]; a = f = None
            fabs = torch.zeros(B, device=dev); aabs = torch.zeros(B, device=dev)
            if self.use_a:
                a = self.aenc(x); feats.append(a); aabs = a.float().abs().amax(-1).view(B, T).amax(1)
                if extras: ex["art"] = self.aenc.residual(x).detach().view(B, T, 9, *x.shape[2:])
            if self.use_f:
                Ft = self.dct(x); f = self.fenc(Ft); feats.append(f); fabs = f.float().abs().amax(-1).view(B, T).amax(1)
                if extras: ex["freq"] = Ft.detach().view(B, T, *Ft.shape[1:])
            aux = torch.stack([self.aux_heads[nm](ft) for nm, ft in zip(self.streams, feats)], 1).float().view(B, T, len(feats), 2)   # per-branch frame logits
            gate3 = torch.zeros(N, 3, device=dev)
            if len(feats) > 1:
                g = torch.softmax(self.gate(torch.cat(feats, -1)).view(N, len(feats), self.D).float(), dim=1).to(s.dtype)   # [N,S,D], sums to 1 over S
                z = (g * torch.stack(feats, 1)).sum(1); gm = g.float().mean(-1).detach()                                        # [N,S]
                for k, nm in enumerate(self.streams): gate3[:, GATE_IDX[nm]] = gm[:, k]
            else: z = s; gate3[:, 0] = 1.0
            z = z.view(B, T, -1)
            if self.use_t:
                v, h = self.temporal(z)
                if extras: ex["attn"] = self.temporal.first_layer_attention(z)
            else: h = z; v = z.mean(1)
            fl = self.frame_head(h).float(); vl = self.video_head(v).float()
        if debug:
            print(f"  Input                [B,T,3,H,W] = {list(x01.shape)}\n  GenConViT            [B,T,D]     = {[B, T, s.shape[-1]]}")
            if a is not None: print(f"  Spatial artifact     [B,T,D]     = {[B, T, a.shape[-1]]}")
            if f is not None: print(f"  (DCT map [B*T,64,h,w] = {list(Ft.shape)})\n  Frequency            [B,T,D]     = {[B, T, f.shape[-1]]}")
            print(f"  Fused                [B,T,D]     = {list(z.shape)}\n  Temporal/frame feat  [B,T,D]     = {list(h.shape)}\n  Video embedding      [B,D]       = {list(v.shape)}\n"
                  f"  Frame logits         [B,T,2]     = {list(fl.shape)}\n  Video logits         [B,2]       = {list(vl.shape)}")
        out = dict(video_logits=vl, frame_logits=fl, rec=rec.float().view(B, T).mean(1), kl=kl.float().view(B, T).mean(1), gate=gate3.view(B, T, 3),
                   freq_absmax=fabs, art_absmax=aabs, aux_logits=aux)
        out.update({"ex_" + k: v_ for k, v_ in ex.items()}); return out

# ================= MODIFIED =================
class TeacherNet(nn.Module):
    """Frozen teacher wrapper. Resizes, calls teacher_forward(self.m, x01, cfg) (user hook), then standardises the output to logits [N,2]."""
    def __init__(self, model, cfg):
        super().__init__(); self.m, self.cfg = model, cfg
    def forward(self, x01):
        S = self.cfg["TEACHER_INPUT_SIZE"]
        if x01.shape[-1] != S: x01 = F.interpolate(x01, size=(S, S), mode="bilinear", align_corners=False)
        o = teacher_forward(self.m, x01, self.cfg).float()
        if o.dim() == 1: o = o.unsqueeze(1)
        if o.shape[1] == 1: o = torch.cat([torch.zeros_like(o), o], 1)
        if self.cfg["TEACHER_FAKE_INDEX"] == 0: o = o.flip(1)
        return o

def load_teacher(cfg, dev):
    path = cfg["TEACHER_CHECKPOINT"]
    if not path or not os.path.exists(path): raise FileNotFoundError(f"TEACHER_CHECKPOINT not found: '{path}' (KD is enabled; it is never silently disabled).")
    log("Teacher: instantiating via build_teacher_model() — make sure it matches YOUR checkpoint architecture (and teacher_forward()).")
    model = build_teacher_model(cfg); ck = load_ckpt(path)
    if isinstance(ck, nn.Module): model = ck
    else:
        sd = ck
        if isinstance(ck, dict):
            for k in ("teacher_state_dict", "model_state_dict", "state_dict", "model", "net", "ema"):
                if k in ck and isinstance(ck[k], dict): sd = ck[k]; break
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}; best = None
        for pre in ("", "model.", "backbone.", "vit.", "teacher.", "m."):
            cand = {(k[len(pre):] if pre and k.startswith(pre) else k): v for k, v in sd.items()}
            res = model.load_state_dict(cand, strict=False); miss = [k for k in res.missing_keys if not k.endswith("num_batches_tracked")]
            if best is None or len(miss) < len(best[1]): best = (pre, miss, res.unexpected_keys, cand)
            if not miss: break
        pre, miss, unexp, cand = best; model.load_state_dict(cand, strict=False)
        tot = len(model.state_dict()); cov = 100.0 * (tot - len(miss)) / max(1, tot)
        log(f"Teacher checkpoint coverage: {cov:.1f}% (missing={len(miss)}, unexpected={len(unexp)}, prefix='{pre}')")
        if unexp: log(f"  unexpected keys (first 15): {unexp[:15]}")
        if miss:
            log(f"  MISSING keys (first 25): {miss[:25]}")
            raise RuntimeError("Teacher architecture does not match the checkpoint (see missing/unexpected keys above). EDIT build_teacher_model(); KD is NOT being disabled.")
    net = TeacherNet(model, cfg).to(dev).eval()
    for p in net.parameters(): p.requires_grad_(False)
    with torch.no_grad(): o = net(torch.rand(2, 3, cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"], device=dev))
    assert o.shape == (2, 2) and torch.isfinite(o).all(), f"Bad teacher output {tuple(o.shape)}"
    log(f"Teacher loaded & FROZEN. params={sum(p.numel() for p in net.parameters())/1e6:.1f}M, output [N,2] OK")
    return net

class TrainWrapper(nn.Module):
    def __init__(self, student, teacher, cfg):
        super().__init__(); self.student, self.teacher, self.amp = student, teacher, cfg["USE_AMP"]
    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None: self.teacher.eval()
        return self
    def forward(self, x01, gen_alpha=1.0, use_teacher=True):
        out = self.student(x01, gen_alpha=gen_alpha)
        if self.teacher is not None and use_teacher:          # ONE teacher pass per batch; frame-level AND video-level KD both derive from out["t_logits"]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                B, T = x01.shape[:2]; out["t_logits"] = self.teacher(x01.flatten(0, 1)).float().view(B, T, 2)
        return out

# ---------------- parameter accounting / summary ----------------
def param_breakdown(student):
    spec = [("GenConViT generative branches (AE/VAE)", ("gen.branches",)), ("Reconstruction-residual CNN", ("gen.resenc",)), ("ConvNeXt/Swin backbone", ("gen.backbones",)), ("GenConViT feature projection", ("gen.proj",)),
            ("Spatial-artifact branch", ("aenc",)), ("Frequency branch (DCT CNN)", ("fenc",)), ("Fusion gate", ("gate",)), ("Temporal branch", ("temporal",)),
            ("Classifier heads (frame+video+aux)", ("frame_head", "video_head", "aux_heads"))]
    rows = []; used = set()
    for name, pre in spec:
        ps = [(n, p) for n, p in student.named_parameters() if n.startswith(pre)]; used.update(n for n, _ in ps)
        rows.append((name, sum(p.numel() for _, p in ps), sum(p.numel() for _, p in ps if p.requires_grad)))
    rest = [(n, p) for n, p in student.named_parameters() if n not in used]
    if rest: rows.append(("other", sum(p.numel() for _, p in rest), sum(p.numel() for _, p in rest if p.requires_grad)))
    return rows

_GF_CACHE = {}
def estimate_gflops(student, cfg, dev):
    """torch FlopCounter GFLOPs (2 x MACs; conv/matmul ops only) for ONE clip [1,T,3,H,W]."""
    if id(student) in _GF_CACHE: return _GF_CACHE[id(student)]
    try:
        from torch.utils.flop_counter import FlopCounterMode
        was = student.training; student.eval(); x = torch.rand(1, cfg["NUM_FRAMES"], 3, cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"], device=dev)
        with torch.no_grad(), FlopCounterMode(display=False) as fc: student(x)
        student.train(was); g = fc.get_total_flops() / 1e9
    except Exception: g = float("nan")
    _GF_CACHE[id(student)] = g; return g

def estimate_teacher_gflops(teacher, cfg, dev):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.rand(cfg["NUM_FRAMES"], 3, cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"], device=dev)
        with torch.no_grad(), FlopCounterMode(display=False) as fc: teacher(x)
        return fc.get_total_flops() / 1e9
    except Exception: return float("nan")

# ================= MODIFIED =================
def print_model_summary(cfg, student, dev):
    on = lambda b: "ON" if b else "OFF"; rows = param_breakdown(student)
    print("=" * 78 + f"\nMODEL: GenConViT-TFKD  ({GENCONVIT_NOTE})\nComponents:")
    print(f"  * GenConViT-based backbone (AE/VAE + {'/'.join(cfg['BACKBONES'])}) = ON   [variant='{cfg['GENCONVIT_VARIANT']}']\n  * Spatial Artifact = {on(cfg['USE_SPATIAL_ARTIFACT'])}\n  * Frequency        = {on(cfg['USE_FREQUENCY'])}\n"
          f"  * Temporal         = {on(cfg['USE_TEMPORAL'])}\n  * KD               = {on(cfg['USE_KD'])}\n  * EMA              = {on(cfg['USE_EMA'])}\nParameters:")
    for n, t, tr in rows: print(f"  {n:42s} {t/1e6:8.3f} M")
    tot = sum(p.numel() for p in student.parameters()); trn = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  {'TOTAL':42s} {tot/1e6:8.3f} M | trainable {trn/1e6:.3f} M (backbone freeze applies only during the first {cfg['FREEZE_BACKBONE_EPOCHS']} epoch(s))")
    print(f"Estimated GFLOPs/clip: {estimate_gflops(student, cfg, dev):.1f}\nInput: [B,{cfg['NUM_FRAMES']},3,{cfg['IMAGE_SIZE']},{cfg['IMAGE_SIZE']}] | Frames/clip: {cfg['NUM_FRAMES']} | "
          f"train window {cfg['TRAIN_WINDOW']} stride {cfg['TEMPORAL_STRIDE']} | eval window {cfg['EVAL_WINDOW']} hop {cfg['EVAL_WINDOW_STRIDE']}\n" + "=" * 78, flush=True)
    return tot, trn

# ================= MODIFIED =================
def cost_summary(cfg, student, teacher, dev):
    """Model cost summary + warning if unusually expensive. NEVER changes the configuration."""
    T = cfg["NUM_FRAMES"]; nb = len(student.gen.branches); imgs = 1 + (nb if student.gen.feed == "recon" else 0); nbk = len(cfg["BACKBONES"])
    tot = sum(p.numel() for p in student.parameters()); trn = sum(p.numel() for p in student.parameters() if p.requires_grad)
    gc = estimate_gflops(student, cfg, dev); gf = gc / T; gt = estimate_teacher_gflops(teacher, cfg, dev) if teacher is not None else float("nan")
    bb_imgs = T * imgs; bb_evals = bb_imgs * nbk; train_est = 3.0 * gc + (gt if math.isfinite(gt) else 0.0)
    print("=" * 78 + "\nMODEL COST SUMMARY (informational; configuration is NOT altered)")
    print(f"  total parameters              : {tot/1e6:.2f} M\n  trainable parameters (initial): {trn/1e6:.2f} M")
    print(f"  approx. GFLOPs / frame        : {gf:.1f}   (student forward, torch FlopCounter = 2 x MACs)\n  approx. GFLOPs / {T}-frame clip  : {gc:.1f}")
    if teacher is not None: print(f"  teacher GFLOPs / clip (frozen): {gt:.1f}")
    print(f"  backbone inputs per clip      : {bb_imgs}  ({T} frames x {imgs} images [original + {imgs - 1} reconstruction(s) fed to the backbones]; GEN_FEED={student.gen.feed})")
    print(f"  backbone forward passes / clip: {bb_evals}  ({bb_imgs} images x {nbk} backbones: {', '.join(cfg['BACKBONES'])})")
    print(f"  approx. training cost / clip  : ~{train_est:.0f} GFLOPs (3 x student fwd+bwd{' + teacher fwd' if teacher is not None else ''})")
    print(f"  EMA enabled: {cfg['USE_EMA']} | KD enabled: {cfg['USE_KD']} | spatial-artifact: {cfg['USE_SPATIAL_ARTIFACT']} | frequency: {cfg['USE_FREQUENCY']} | temporal: {cfg['USE_TEMPORAL']}")
    print(f"  GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)} | DataParallel={cfg['USE_DATAPARALLEL']} | AMP={cfg['USE_AMP']}")
    if (math.isfinite(gc) and gc > 300) or bb_evals > 32:
        print(f"  !! WARNING: this configuration is UNUSUALLY EXPENSIVE for 2 x Tesla T4 (estimated {bb_evals} backbone passes and {gc:.0f} GFLOPs per clip, plus "
              f"EMA{' + frozen teacher' if teacher is not None else ''} + spatial/frequency branches + temporal transformer).\n"
              f"     Expect long epochs. Options YOU may consider (nothing is changed automatically): GENCONVIT_VARIANT='ed' ({T*2*nbk} backbone passes/clip), "
              f"fewer NUM_FRAMES, TRAIN_CLIPS_PER_EPOCH, or USE_EMA=False. GENCONVIT_VARIANT='both' remains fully supported.")
    print("=" * 78, flush=True)
    return dict(params=tot, trainable=trn, gflops_clip=gc, gflops_frame=gf, gflops_teacher=gt, backbone_passes=bb_evals)

def write_ablation_csv(cfg, R, tag):
    pd.DataFrame([dict(experiment=R["exp"], model_tag=tag, genconvit=True, genconvit_variant=cfg["GENCONVIT_VARIANT"], genconvit_implementation=GENCONVIT_NOTE,
                       spatial_artifact=cfg["USE_SPATIAL_ARTIFACT"], frequency=cfg["USE_FREQUENCY"], temporal=cfg["USE_TEMPORAL"], kd=cfg["USE_KD"], ema=cfg["USE_EMA"],
                       num_frames=cfg["NUM_FRAMES"], train_window=cfg["TRAIN_WINDOW"], temporal_stride=cfg["TEMPORAL_STRIDE"], eval_window=cfg["EVAL_WINDOW"], eval_hop=cfg["EVAL_WINDOW_STRIDE"],
                       lambda_kd=cfg["LAMBDA_KD"], temperature=cfg["TEMPERATURE"], lambda_tc=cfg["LAMBDA_TC"], lambda_rec=cfg["LAMBDA_REC"], lambda_kl=cfg["LAMBDA_KL"],
                       generative_warmup_epochs=cfg["GENERATIVE_WARMUP_EPOCHS"], generative_input_warmup_epochs=cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"], video_aggregation=cfg["VIDEO_AGGREGATION"])]).to_csv(os.path.join(R["metrics"], "ablation_configuration.csv"), index=False)

# =====================================================================================
#                                  LOSSES & METRICS
# =====================================================================================
def kd_loss(s_logits, t_prob, tau):
    """tau^2 * KL(p_T || p_S). t_prob is the teacher's already temperature-softened distribution [N,2]."""
    return F.kl_div(F.log_softmax(s_logits.float() / tau, -1), t_prob.float(), reduction="batchmean") * tau * tau

def kd_per_sample(s_logits, t_prob, tau):
    return F.kl_div(F.log_softmax(s_logits.float() / tau, -1), t_prob.float(), reduction="none").sum(-1) * tau * tau
    
def js_adjacent(fl):
    p = F.softmax(fl.float(), -1).clamp_min(1e-7); p1, p2 = p[:, :-1], p[:, 1:]; m = 0.5 * (p1 + p2)
    kl = lambda a, b: (a * (a.log() - b.log())).sum(-1)
    return (0.5 * kl(p1, m) + 0.5 * kl(p2, m)).mean()

# ================= MODIFIED =================
def gen_scale(cfg, global_step, steps_per_epoch):
    """Linear warm-up factor in [0,1] for the generative losses (L_REC, L_KL): 0 -> 1 over GENERATIVE_WARMUP_EPOCHS epochs (per update step)."""
    W = cfg["GENERATIVE_WARMUP_EPOCHS"]
    if W <= 0: return 1.0
    return float(min(1.0, global_step / max(1.0, W * steps_per_epoch)))

# ================= MODIFIED =================
def gen_input_alpha(cfg, global_step, steps_per_epoch):
    """Generative-INPUT warm-up factor alpha in [0,1] (NOT the loss-weight ramp gen_scale): backbone input = (1-alpha)*original + alpha*reconstruction.
    alpha = 0 at global_step 0 and reaches 1 after GENERATIVE_INPUT_WARMUP_EPOCHS * steps_per_epoch update steps. Depends ONLY on (global_step, steps_per_epoch)
    -> a resumed run (global_step restored from the checkpoint) continues exactly the same schedule. 0 epochs = off (alpha = 1)."""
    W = cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"]
    if W <= 0: return 1.0
    return float(min(1.0, max(0.0, global_step / max(1.0, W * steps_per_epoch))))

def kd_scale(cfg, global_step, steps_per_epoch):
    """KD weight factor in [0,1]: 0 until KD_START_EPOCH, then a linear ramp over KD_RAMP_EPOCHS (depends only on global_step -> resume-safe)."""
    s0 = cfg["KD_START_EPOCH"] * steps_per_epoch; r = max(1.0, cfg["KD_RAMP_EPOCHS"] * steps_per_epoch)
    return float(min(1.0, max(0.0, (global_step - s0) / r)))

def _norm_entropy(p): p = p.clamp_min(1e-7); return -(p * p.log()).sum(-1) / math.log(2.0)

def compute_loss(out, y, cfg, gscale=1.0, kscale=1.0):
    """L = l_CE*L_CE + l_KD*L_KD + l_TC*L_TC + l_REC*w*L_REC + l_KL*w*L_KL, w = generative warm-up factor (1.0 after warm-up)."""
    vl, fl = out["video_logits"], out["frame_logits"]; B, T, _ = fl.shape; ls = cfg["LABEL_SMOOTHING"]
    ce = F.cross_entropy(vl, y, label_smoothing=ls) + cfg["FRAME_CE_WEIGHT"] * F.cross_entropy(fl.reshape(B * T, 2), y.repeat_interleave(T), label_smoothing=ls)
    z = torch.zeros((), device=vl.device); kd = z; tc = z; kd_cov = z
    if cfg["USE_KD"] and "t_logits" in out:
        tau = cfg["TEMPERATURE"]; tl = out["t_logits"].float(); pt_frame = F.softmax(tl / tau, -1); pt_video = pt_frame.mean(1)
        wmode = cfg.get("KD_WEIGHTING", "confidence"); p_t = F.softmax(tl, -1); p_tv = p_t.mean(1); yf = y.view(B, 1).expand(B, T)
        cor_f = (tl.argmax(-1) == yf).float().reshape(-1); cor_v = (p_tv.argmax(-1) == y).float()
        if wmode == "correct_only": ok_f, ok_v = cor_f, cor_v                                              # hard mask: teacher must be right
        elif wmode == "confidence": ok_f = p_t.gather(-1, yf.unsqueeze(-1)).squeeze(-1).reshape(-1); ok_v = p_tv.gather(-1, y.view(B, 1)).squeeze(-1)   # teacher prob of the TRUE class
        elif wmode == "entropy": ok_f = (1.0 - _norm_entropy(p_t)).reshape(-1) * cor_f; ok_v = (1.0 - _norm_entropy(p_tv)) * cor_v                       # confident AND correct
        elif wmode == "none": ok_f = torch.ones(B * T, device=vl.device); ok_v = torch.ones(B, device=vl.device)
        else: raise ValueError(f"KD_WEIGHTING must be correct_only | confidence | entropy | none, got '{wmode}'")
        kd_cov = (ok_f > 0.5).float().mean()
        kd_f = (kd_per_sample(fl.reshape(B * T, 2), pt_frame.reshape(B * T, 2), tau) * ok_f).sum() / ok_f.sum().clamp_min(1.0)
        kd_v = (kd_per_sample(vl, pt_video, tau) * ok_v).sum() / ok_v.sum().clamp_min(1.0)
        kd = 0.5 * (kd_f + kd_v)
    if cfg["USE_TEMPORAL"] and cfg["LAMBDA_TC"] > 0 and T > 1: tc = js_adjacent(fl)
    rec = out["rec"].float().mean(); klv = out["kl"].float().mean()
    w_rec = cfg["LAMBDA_REC"] * gscale; w_kl = cfg["LAMBDA_KL"] * gscale
    aux = z
    if "aux_logits" in out and cfg.get("AUX_CE_WEIGHT", 0) > 0:
        al = out["aux_logits"].float(); S_ = al.shape[2]
        aux = F.cross_entropy(al.reshape(B * T * S_, 2), y.view(B, 1).expand(B, T * S_).reshape(-1), label_smoothing=ls)
    total = cfg["LAMBDA_CE"] * ce + cfg["LAMBDA_KD"] * kscale * kd + cfg["LAMBDA_TC"] * tc + w_rec * rec + w_kl * klv + cfg.get("AUX_CE_WEIGHT", 0.0) * aux
    return dict(total=total, ce=ce, kd=kd, tc=tc, rec=rec, aux=aux, kd_cov=kd_cov)

def nonfinite_report(out, L):
    return [k for k, v in list(out.items()) + list(L.items()) if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]

# ================= MODIFIED =================
def reliability_bins(y, p, thr=0.5, bins=15):
    """Standard binary reliability bins. confidence = max(p,1-p); prediction = (p>=thr); correct = (prediction==y).
    Returns [(mean_confidence, empirical_accuracy, count), ...] over equal-width confidence bins in [0.5,1]. Used by BOTH ECE and the figure."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1 - p); correct = ((p >= thr).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1); rows = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any(): rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows

def ece_score(y, p, thr=0.5, bins=15):
    n = len(y)
    if n == 0: return float("nan")
    return float(sum(c / n * abs(a - cf) for cf, a, c in reliability_bins(y, p, thr, bins)))

def youden_threshold(y, p):
    if len(np.unique(y)) < 2: return 0.5
    fpr, tpr, thr = roc_curve(y, p); ok = np.isfinite(thr); j = (tpr - fpr)[ok]
    return float(thr[ok][int(np.argmax(j))])

def compute_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float); pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel(); two = len(np.unique(y)) == 2
    rec = tp / max(1, tp + fn); spec = tn / max(1, tn + fp); prec = tp / max(1, tp + fp)
    m = dict(thr=thr, acc=(tp + tn) / len(y), bal_acc=0.5 * (rec + spec), precision=prec, recall=rec, specificity=spec,
             f1=2 * prec * rec / max(1e-12, prec + rec), mcc=float(matthews_corrcoef(y, pred)) if two else float("nan"),
             auc=float(roc_auc_score(y, p)) if two else float("nan"), pr_auc=float(average_precision_score(y, p)) if two else float("nan"),
             brier=float(np.mean((p - y) ** 2)), ece=ece_score(y, p, thr), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=len(y))
    if two:
        fpr, tpr, _ = roc_curve(y, p); fnr = 1 - tpr; i = int(np.argmin(np.abs(fnr - fpr))); m["eer"] = float((fpr[i] + fnr[i]) / 2)
        m["nll"] = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    else: m["eer"] = m["nll"] = float("nan")
    return m

AGG_RULES = ("mean", "median", "max", "topk_mean", "soft_topk", "conf_weighted")
def aggregate(ps, rule):
    ps = np.asarray(ps, dtype=float)
    if rule == "median": return float(np.median(ps))
    if rule == "max": return float(np.max(ps))
    if rule == "topk_mean": k = max(1, len(ps) // 2); return float(np.sort(ps)[-k:].mean())
    if rule == "soft_topk": w = np.exp(5.0 * ps); return float((w * ps).sum() / w.sum())                 # softmax-weighted: leans toward the most 'fake' clips
    if rule == "conf_weighted": w = np.abs(ps - 0.5) + 1e-3; return float((w * ps).sum() / w.sum())       # confident clips count more
    return float(ps.mean())

# =====================================================================================
#                                     INFERENCE
# =====================================================================================
@torch.no_grad()
def run_inference(model, ds, cfg, desc, collect_frames=True):
    model.eval(); dev = torch.device("cuda")
    gc.collect()
    dl = DataLoader(ds, batch_size=cfg["EVAL_BATCH_SIZE"], shuffle=False, num_workers=cfg.get("NUM_WORKERS_EVAL", cfg["NUM_WORKERS"]), pin_memory=True)
    crow, frow = [], []
    it = iter(dl)
    for x, y, idx in tqdm(it, total=len(dl), desc=desc, leave=False):
        x01 = to_x01(x, dev); out = model(x01); yd = y.to(dev)
        if cfg.get("TTA_FLIP", False):
            o2 = model(x01.flip(-1))
            for k_ in ("video_logits", "frame_logits", "gate"): out[k_] = 0.5 * (out[k_].float() + o2[k_].float())
        pv = F.softmax(out["video_logits"].float(), -1)[:, 1].cpu().numpy(); pf = F.softmax(out["frame_logits"].float(), -1)[..., 1].cpu().numpy()
        gate = out["gate"].float().mean(1).cpu().numpy(); loss = F.cross_entropy(out["video_logits"].float(), yd, reduction="none").cpu().numpy()
        for b, i in enumerate(idx.tolist()):
            v, fr = ds.meta(i)
            crow.append((v["video_id"], int(v["label"]), v["generator"], float(pv[b]), float(pf[b].mean()), float(gate[b, 0]), float(gate[b, 1]), float(gate[b, 2]), float(loss[b])))
            if collect_frames:
                for t, fn in enumerate(fr): frow.append((v["video_id"], fn, int(v["label"]), v["generator"], float(pf[b, t])))
    close_iter(it); del it, dl
    c = pd.DataFrame(crow, columns=["video_id", "label", "generator", "p_clip", "p_frame_mean", "gs", "ga", "gf", "loss"])
    vdf = c.groupby("video_id", sort=False).agg(label=("label", "first"), generator=("generator", "first"),
            p_video=("p_clip", lambda s: aggregate(s.values, cfg["VIDEO_AGGREGATION"])), p_frame_mean=("p_frame_mean", "mean"),
            gate_spatial=("gs", "mean"), gate_artifact=("ga", "mean"), gate_frequency=("gf", "mean"), n_clips=("p_clip", "size")).reset_index()
    for r_ in AGG_RULES:                                    # one extra column per aggregation rule -> validation can compare them (selection on VAL only)
        vdf["p_" + r_] = c.groupby("video_id", sort=False)["p_clip"].apply(lambda s, r=r_: aggregate(s.values, r)).reindex(vdf["video_id"]).values
    fdf = None
    if collect_frames:
        fdf = pd.DataFrame(frow, columns=["video_id", "frame_id", "label", "generator", "p_frame"]).groupby(["video_id", "frame_id"], sort=False).agg(
            label=("label", "first"), generator=("generator", "first"), p_frame=("p_frame", "mean")).reset_index()
    return fdf, vdf, float(c["loss"].mean())

def fmt_agg(v): return ", ".join(f"{k}={a:.3f}" for k, a in v.get("agg_auc", {}).items())

def validate(model, ds, cfg):
    _, vdf, loss = run_inference(model, ds, cfg, "val", collect_frames=False)
    y = vdf["label"].values; p = vdf["p_video"].values
    m = compute_metrics(y, p, 0.5); m["loss"] = loss
    thr = youden_threshold(y, p); mo = compute_metrics(y, p, thr)        # optimistic (threshold fitted on the same val set): for monitoring only
    m.update(thr_opt=thr, f1_opt=mo["f1"], bal_acc_opt=mo["bal_acc"], mcc_opt=mo["mcc"])
    if cfg.get("VAL_AGG_SWEEP", True) and len(np.unique(y)) == 2: m["agg_auc"] = {r: float(roc_auc_score(y, vdf["p_" + r])) for r in AGG_RULES}
    if len(np.unique(y)) == 2:                                   # per-manipulation AUC: all real videos vs the fakes of ONE generator
        real = vdf[vdf["label"] == 0]; ga = {}
        for g_ in sorted(set(vdf[vdf["label"] == 1]["generator"])):
            sub = pd.concat([real, vdf[(vdf["label"] == 1) & (vdf["generator"] == g_)]])
            ga[g_] = float(roc_auc_score(sub["label"].values, sub["p_video"].values))
        m["gen_auc"] = ga
    return m

@torch.no_grad()
def eval_teacher(teacher, ds, cfg):
    dev = torch.device("cuda"); dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=cfg["NUM_WORKERS"], pin_memory=True)
    pf, pv, yy = [], [], []
    for x, y, _ in tqdm(dl, desc="teacher-val", leave=False):
        x01 = to_x01(x, dev); B, T = x01.shape[:2]
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["USE_AMP"]): tl = teacher(x01.flatten(0, 1)).float().view(B, T, 2)
        p = F.softmax(tl, -1)[..., 1]; pf.append(p.flatten().cpu().numpy()); pv.append(p.mean(1).cpu().numpy()); yy.append(y.numpy())
    y = np.concatenate(yy); return compute_metrics(y, np.concatenate(pv), 0.5), compute_metrics(np.repeat(y, cfg["NUM_FRAMES"]), np.concatenate(pf), 0.5)

# =====================================================================================
#                                 EMA / OPTIM / CHECKPOINT
# =====================================================================================
class EMA:
    def __init__(self, model, decay): self.m = copy.deepcopy(model).eval(); self.decay = decay; self.steps = 0; [p.requires_grad_(False) for p in self.m.parameters()]
    @torch.no_grad()
    def update(self, model):
        self.steps += 1; d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        ep, mp = list(self.m.parameters()), [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ep, d); torch._foreach_add_(ep, mp, alpha=1 - d)
        for be, bm in zip(self.m.buffers(), model.buffers()): be.copy_(bm)
    def state_dict(self): return {"model": self.m.state_dict(), "steps": self.steps}
    def load_state_dict(self, sd): self.m.load_state_dict(sd["model"]); self.steps = sd["steps"]

def param_group_name(n):
    if n.startswith("gen.backbones."): return "backbone"
    if n.startswith("gen.branches."): return "generative"
    if n.startswith(("aenc.", "fenc.", "gen.resenc.")): return "forensic"
    if n.startswith("temporal."): return "temporal"
    if n.startswith(("gate.", "gen.proj.")): return "fusion"
    return "heads"

def make_optimizer(student, cfg):
    lr_of = {"backbone": cfg["LR_BACKBONE"], "generative": cfg["LR_GENERATIVE"], "forensic": cfg["LR_FORENSIC"], "temporal": cfg["LR_TEMPORAL"], "fusion": cfg["LR_FUSION"], "heads": cfg["LR_HEADS"]}
    g = defaultdict(list)
    for n, p in student.named_parameters():
        dec = p.ndim > 1 and ("temporal.cls" not in n) and ("temporal.pos" not in n)
        g[(param_group_name(n), dec)].append(p)
    groups = [dict(params=ps, name=f"{k}_{'decay' if d else 'nodecay'}", lr=lr_of[k], weight_decay=cfg["WEIGHT_DECAY"] if d else 0.0) for (k, d), ps in g.items()]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))

def arch_keys(): return ["USE_TEMPORAL", "USE_FREQUENCY", "USE_SPATIAL_ARTIFACT", "NUM_FRAMES", "IMAGE_SIZE", "EMBED_DIM", "GENCONVIT_VARIANT", "BACKBONES", "TEMP_LAYERS", "TEMP_HEADS", "FREQ_TYPE"]

def cfg_to_jsonable(cfg): return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}

# ================= MODIFIED =================
def gen_input_state(S):
    spe = getattr(S, "steps_per_epoch", None)
    return dict(warmup_epochs=S.cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"], steps_per_epoch=spe, global_step=S.global_step, alpha=gen_input_alpha(S.cfg, S.global_step, spe) if spe else None)

def check_gen_input(ck, cfg, steps_per_epoch):
    """Resume consistency for the generative-input warm-up: a different GENERATIVE_INPUT_WARMUP_EPOCHS would change the schedule -> refuse. (A checkpoint that predates it only triggers the WARN loop.)"""
    a = ck["config"].get("GENERATIVE_INPUT_WARMUP_EPOCHS")
    if a is not None and a != cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"]:
        raise RuntimeError(f"Resume mismatch: GENERATIVE_INPUT_WARMUP_EPOCHS ckpt={a} vs CONFIG={cfg['GENERATIVE_INPUT_WARMUP_EPOCHS']} (the generative-input alpha schedule would not continue identically).")
    g = ck.get("gen_input") or {}
    if g.get("steps_per_epoch") not in (None, steps_per_epoch): log(f"WARN: steps_per_epoch for the generative-input schedule changed ({g['steps_per_epoch']} -> {steps_per_epoch}); alpha(global_step) will differ from the original schedule.")

def save_full_ckpt(path, S, partial=None, tag=""):
    ck = dict(model=S.student.state_dict(), optimizer=S.opt.state_dict(), scheduler=S.sched.state_dict(), scaler=S.scaler.state_dict(),
              ema=S.ema.state_dict() if S.ema else None, epoch=S.epochs_done, partial=partial, global_step=S.global_step, update_step=S.global_step,
              best=S.best, history=S.hist, config=cfg_to_jsonable(S.cfg), rng=get_rng_state(), split_signature=S.sig, model_tag=model_tag(S.cfg),
              finished=S.finished, tag=tag, torch=torch.__version__, gen_input=gen_input_state(S))
    atomic_save(ck, path)

# ================= MODIFIED =================
def check_arch(ck_cfg, cfg):
    for k in arch_keys():
        a, b = ck_cfg.get(k), cfg[k]; a = tuple(a) if isinstance(a, list) else a; b = tuple(b) if isinstance(b, list) else b
        if a != b: raise ValueError(f"Resume mismatch on architecture key {k}: ckpt={a} vs CONFIG={b}")

class StratifiedPlan:
    """Deterministic epoch plan: real/fake balance AND an equal quota per manipulation folder. Each group consumes a long reshuffled stream across epochs, so a video
    repeats as late as possible. Depends only on (seed, epoch) -> exact mid-epoch resume."""
    def __init__(self, groups, real_frac=0.5):             # groups: {name: (label, [item indices])}
        self.g = sorted(groups.items()); nr = sum(1 for _, (lab, _) in self.g if lab == 0); nf = len(self.g) - nr
        self.frac = [real_frac / nr if lab == 0 else (1 - real_frac) / nf for _, (lab, _) in self.g]
    @staticmethod
    def _stream(items, seed, gid, upto):
        out = []; c = 0
        while len(out) < upto: out += [items[j] for j in np.random.default_rng([seed, gid, c]).permutation(len(items))]; c += 1
        return out[:upto]
    def draw(self, n, seed, epoch):
        quotas = [int(round(n * f)) for f in self.frac]; quotas[0] += n - sum(quotas); keys = []
        for gid, ((name, (lab, items)), q) in enumerate(zip(self.g, quotas)): keys += self._stream(items, seed, gid, (epoch + 1) * q)[epoch * q:]
        order = np.random.default_rng([seed, 99991, epoch]).permutation(len(keys))
        return list(enumerate([int(keys[i]) for i in order]))

def sampler_report(keys, ds, vids, epoch):
    items = [k[1] for k in keys]; per = defaultdict(int)
    for i in items: per[vids[ds.items[i][0]]["generator"]] += 1
    nreal = sum(c for g, c in per.items() if any(r in g.lower() for r in REAL_KEYS))
    log(f"Epoch {epoch+1} sampler: {len(items)} clips | unique videos {len(set(items))} (repeat rate {1 - len(set(items)) / len(items):.1%}) | real {nreal} fake {len(items) - nreal} | per folder {dict(per)}")

def epoch_keys(w_t, n_samples, seed, epoch):
    """Deterministic (draw_position, item_index) list for an epoch; identical for a given (seed, epoch). Positions are GLOBAL within the epoch,
    so a mid-epoch resume (slicing this list) reproduces exactly the clips the uninterrupted run would have seen."""
    if hasattr(w_t, "draw"): return w_t.draw(n_samples, seed, epoch)
    g = torch.Generator().manual_seed(seed + epoch)
    return list(enumerate(torch.multinomial(w_t, n_samples, replacement=True, generator=g).tolist()))

def plan_resume(ck, steps_per_epoch, accum, bs):
    """Partial checkpoint (epoch=E, completed update steps=S) -> training continues at update step S+1 of epoch E (index E)."""
    p = ck.get("partial")
    if not p: return dict(epoch_index=int(ck["epoch"]), start_step=0, next_update_step=1, skip_micro=0, skip_samples=0, acc=None)
    if int(p["epoch"]) != int(ck["epoch"]): raise RuntimeError(f"Inconsistent partial checkpoint (partial epoch {p['epoch']} vs epochs_done {ck['epoch']}).")
    if p.get("steps_per_epoch") is not None and int(p["steps_per_epoch"]) != int(steps_per_epoch):
        raise RuntimeError(f"steps_per_epoch changed since the partial checkpoint ({p['steps_per_epoch']} -> {steps_per_epoch}): batch/accum/dataset configuration differs; cannot resume mid-epoch.")
    s = int(p["step_in_epoch"])
    if not 0 <= s < steps_per_epoch: raise RuntimeError(f"Invalid partial step {s} (steps_per_epoch={steps_per_epoch}).")
    return dict(epoch_index=int(ck["epoch"]), start_step=s, next_update_step=s + 1, skip_micro=s * accum, skip_samples=s * accum * bs, acc=p.get("acc"))

def resume_selftest(cfg, R, S, steps_per_epoch, accum, bs, w_t, n_samples):
    """Checkpoint/resume self-test (no training): save a temporary checkpoint, reload it and verify every restored quantity + resume position."""
    log("Checkpoint/resume self-test ...")
    E = 2; Sx = max(0, min(17, steps_per_epoch - 1)); T = SimpleNamespace(**vars(S))
    T.epochs_done = E; T.global_step = E * steps_per_epoch + Sx; T.finished = False
    T.best = dict(auc=0.9123, f1=0.8, bal_acc=0.85, epoch=E, source="ema"); T.hist = [dict(epoch=i + 1, val_auc=0.9) for i in range(E)]
    partial = dict(epoch=E, step_in_epoch=Sx, micro_in_epoch=Sx * accum, steps_per_epoch=steps_per_epoch, acc={"n": 0.0})
    tmp = os.path.join(R["ckpt"], "_resume_selftest.pth"); save_full_ckpt(tmp, T, partial=partial, tag="selftest"); ck = load_ckpt(tmp); os.remove(tmp)
    assert ck["epoch"] == E and ck["update_step"] == T.global_step and ck["global_step"] == T.global_step, "epoch/update step mismatch"
    assert ck["partial"]["step_in_epoch"] == Sx and ck["partial"]["epoch"] == E, "partial step mismatch"
    assert abs(ck["best"]["auc"] - 0.9123) < 1e-12, "best AUC mismatch"
    sd = S.student.state_dict(); assert set(ck["model"]) == set(sd), "model keys mismatch"
    assert all(torch.equal(ck["model"][k].cpu(), v.detach().cpu()) for k, v in sd.items()), "model weights mismatch"
    oa, ob = ck["optimizer"], S.opt.state_dict(); assert len(oa["param_groups"]) == len(ob["param_groups"]), "optimizer group count mismatch"
    for ga, gb in zip(oa["param_groups"], ob["param_groups"]):
        for key in ("lr", "weight_decay", "params"): assert ga[key] == gb[key], f"optimizer group '{key}' mismatch"
    assert set(oa["state"]) == set(ob["state"]), "optimizer state keys mismatch"
    for k in ob["state"]:
        for kk, vv in ob["state"][k].items():
            if torch.is_tensor(vv): assert torch.equal(oa["state"][k][kk].cpu(), vv.cpu()), "optimizer state tensor mismatch"
    assert ck["scheduler"]["last_epoch"] == S.sched.state_dict()["last_epoch"], "scheduler state mismatch"
    assert ck["scaler"] == S.scaler.state_dict(), "GradScaler state mismatch"
    if S.ema is not None: assert ck["ema"]["steps"] == S.ema.steps, "EMA state mismatch"
    check_arch(ck["config"], cfg); check_gen_input(ck, cfg, steps_per_epoch)
    gi = ck["gen_input"]; assert gi["warmup_epochs"] == cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"] and gi["steps_per_epoch"] == steps_per_epoch and gi["global_step"] == T.global_step, "generative-input state mismatch"
    assert gi["alpha"] == gen_input_alpha(cfg, T.global_step, steps_per_epoch) == gen_input_alpha(ck["config"], T.global_step, steps_per_epoch), "generative-input alpha does not resume identically"
    Wg = cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"]; al = [gen_input_alpha(cfg, s_, steps_per_epoch) for s_ in np.linspace(0, max(1, Wg) * steps_per_epoch * 1.5, 25).astype(int)]
    assert all(0.0 <= a_ <= 1.0 for a_ in al) and all(y_ >= x_ for x_, y_ in zip(al, al[1:])), "generative-input alpha must be monotone in [0,1]"
    if Wg > 0: assert gen_input_alpha(cfg, 0, steps_per_epoch) == 0.0 and gen_input_alpha(cfg, Wg * steps_per_epoch, steps_per_epoch) == 1.0 and gen_input_alpha(cfg, (Wg + 3) * steps_per_epoch, steps_per_epoch) == 1.0, "alpha endpoints wrong"
    plan = plan_resume(ck, steps_per_epoch, accum, bs)
    assert plan["epoch_index"] == E and plan["start_step"] == Sx and plan["next_update_step"] == Sx + 1, "resume must continue at step S+1"
    keys = epoch_keys(w_t, n_samples, cfg["SEED"], E); sub = keys[plan["skip_micro"] * bs: steps_per_epoch * accum * bs]
    assert sub[0][0] == Sx * accum * bs, "resumed sampler does not start at the first unprocessed sample"
    assert len(sub) // bs == (steps_per_epoch - Sx) * accum, "resumed epoch has the wrong number of micro-batches"
    assert keys == epoch_keys(w_t, n_samples, cfg["SEED"], E), "epoch order is not deterministic"
    log(f"  self-test OK: saved epoch={E}, update step={T.global_step}, partial step={Sx}, best AUC=0.9123; model/optimizer/scheduler/scaler"
        f"{'/EMA' if S.ema is not None else ''}/architecture + generative-input warm-up schedule verified; partial (epoch {E}, step {Sx}) resumes at update step {Sx+1} (skips {plan['skip_micro']} micro-batches), not step 1.")

# =====================================================================================
#                                  PLOTTING HELPERS
# =====================================================================================
def setup_mpl():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
def savefig(fig, base):
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight"); fig.savefig(base + ".pdf", bbox_inches="tight"); plt.close(fig)

def plot_curve(kind, series, title, base):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for lab, (y, p) in series.items():
        if len(np.unique(y)) < 2: continue
        if kind == "roc": fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, lw=1.8, label=f"{lab} (AUC {roc_auc_score(y, p):.3f})")
        else: pr, rc, _ = precision_recall_curve(y, p); ax.plot(rc, pr, lw=1.8, label=f"{lab} (AP {average_precision_score(y, p):.3f})")
    if kind == "roc": ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    else: ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_title(title); ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8); savefig(fig, base)

def plot_cm(m, title, base):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]]); fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.grid(False)
    ax.imshow(cm / np.maximum(1, cm.sum(1, keepdims=True)), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i, j]}\n({100*cm[i, j]/max(1, cm[i].sum()):.1f}%)", ha="center", va="center", fontsize=9, color="black" if cm[i, j] / max(1, cm[i].sum()) < 0.6 else "white")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title); savefig(fig, base)

def plot_training(hist, fig_dir):
    if not hist: return
    h = pd.DataFrame(hist); e = h["epoch"]
    for key, cols, yl, name in (("loss", [("train_loss", "Train"), ("val_loss", "Val")], "Loss", "training_loss_curve"),
                               ("auc", [("val_auc", "Val AUC")], "ROC-AUC", "validation_auc_curve"), ("f1", [("val_f1", "Val F1")], "F1", "validation_f1_curve")):
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for c, l in cols:
            if c in h: ax.plot(e, h[c], marker="o", ms=3, lw=1.6, label=l)
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.legend(); savefig(fig, os.path.join(fig_dir, name))

# ================= MODIFIED =================
def plot_reliability(res_items, ece_of, base, bins=15):
    """Conventional binary reliability diagram: x = mean confidence max(p,1-p), y = empirical accuracy of (p>=thr), diagonal = perfect calibration.
    Uses the same reliability_bins() and the same threshold as the ECE reported in final_metrics_summary.csv."""
    fig, axs = plt.subplots(1, len(res_items), figsize=(3.4 * len(res_items), 3.5), squeeze=False)
    for a_, (name, y, p, thr) in zip(axs[0], res_items):
        rows = reliability_bins(y, p, thr, bins); a_.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=0.9, label="Perfect calibration")
        if rows:
            cf = [r[0] for r in rows]; ac = [r[1] for r in rows]; cnt = np.array([r[2] for r in rows], dtype=float)
            a_.plot(cf, ac, "-", lw=1.2, color="tab:blue"); a_.scatter(cf, ac, s=12 + 70 * np.sqrt(cnt / cnt.max()), color="tab:blue", zorder=3, label="Model (marker size ~ #videos)")
        a_.set_xlim(0.5, 1.0); a_.set_ylim(0.0, 1.02); a_.set_xlabel("Mean confidence  max(p, 1-p)"); a_.set_title(f"{name}\nECE = {ece_of[name]:.3f} (thr {thr:.3f})", fontsize=9)
    axs[0, 0].set_ylabel("Empirical accuracy"); axs[0, 0].legend(loc="lower right", fontsize=7); savefig(fig, base)

def bootstrap_ci(y, p, thr, iters, seed):
    rng = np.random.default_rng(seed); n = len(y); res = defaultdict(list)       # resamples VIDEOS (one row per video)
    for _ in range(iters):
        idx = rng.integers(0, n, n); yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2: continue
        pred = (pp >= thr).astype(int); tp = ((pred == 1) & (yy == 1)).sum(); tn = ((pred == 0) & (yy == 0)).sum(); fp = ((pred == 1) & (yy == 0)).sum(); fn = ((pred == 0) & (yy == 1)).sum()
        rc = tp / max(1, tp + fn); sp = tn / max(1, tn + fp); pr = tp / max(1, tp + fp)
        res["auc"].append(roc_auc_score(yy, pp)); res["f1"].append(2 * pr * rc / max(1e-12, pr + rc)); res["acc"].append((tp + tn) / n); res["bal_acc"].append(0.5 * (rc + sp))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)) for k, v in res.items()}

# =====================================================================================
#                          PREFLIGHT  (sanity checks + batch-size probe)
# =====================================================================================
def loss_gradient_diagnostic(wrapper, student, train_ds, cfg, dev):
    """Lightweight diagnostic, NO optimizer step: (1) teacher confidence/correctness/KD-coverage on 48 balanced TRAIN clips (teacher forward only);
    (2) weighted loss shares, per-loss gradient norm per module group, fusion-gate entropy on 2 clips. Failure is logged loudly but does not abort training."""
    log("---------- LOSS / GRADIENT / TEACHER DIAGNOSTIC (tiny batches, no optimizer step) ----------")
    rs = get_rng_state(); was = wrapper.training
    try:
        train_ds.set_epoch(0); wrapper.train(); rng = np.random.default_rng(0); T = cfg["NUM_FRAMES"]
        labs = np.array([train_ds.v[vi]["label"] for vi, _, _ in train_ds.items])
        if wrapper.teacher is not None:
            ids = [int(i) for i in rng.choice(np.where(labs == 0)[0], 24, replace=False)] + [int(i) for i in rng.choice(np.where(labs == 1)[0], 24, replace=False)]
            P, Y = [], []
            with torch.no_grad():
                for b in range(0, len(ids), 4):
                    chunk = [train_ds[(b + j, i)] for j, i in enumerate(ids[b:b + 4])]
                    x01 = to_x01(torch.stack([c[0] for c in chunk]), dev)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["USE_AMP"]): tl = wrapper.teacher(x01.flatten(0, 1)).float()
                    P.append(F.softmax(tl, -1)[:, 1].cpu().view(-1, T)); Y.append(torch.tensor([int(c[1]) for c in chunk]))
            P = torch.cat(P); Y = torch.cat(Y); tau = cfg["TEMPERATURE"]
            for cls, nm in ((0, "REAL"), (1, "FAKE")):
                p = P[Y == cls].clamp(1e-6, 1 - 1e-6); ptrue = p if cls == 1 else 1 - p
                zl = torch.log(p / (1 - p)); pt = torch.sigmoid(zl / tau)
                ent = -(pt * pt.log() + (1 - pt) * (1 - pt).log()) / math.log(2.0)
                print(f"  teacher on TRAIN {nm}: frame-correct={float((ptrue > 0.5).float().mean()):.3f} | mean conf={float(torch.maximum(p, 1 - p).mean()):.3f} | frames with conf>0.9: {float((torch.maximum(p, 1 - p) > 0.9).float().mean()):.3f} | "
                      f"mean p(true class)={float(ptrue.mean()):.3f} | KD-target entropy @tau={tau}: {float(ent.mean()):.3f} (1=uninformative) | KD coverage (correct_only)={float((ptrue > 0.5).float().mean()):.3f}", flush=True)
        pick = [int(rng.choice(np.where(labs == 0)[0])), int(rng.choice(np.where(labs == 1)[0]))]
        x01 = to_x01(torch.stack([train_ds[(k, i)][0] for k, i in enumerate(pick)]), dev); y = torch.tensor([0, 1], device=dev)
        out = wrapper(x01, gen_alpha=1.0); L = compute_loss(out, y, cfg, 1.0, 1.0)
        terms = dict(ce=cfg["LAMBDA_CE"] * L["ce"], kd=cfg["LAMBDA_KD"] * L["kd"], tc=cfg["LAMBDA_TC"] * L["tc"], rec=cfg["LAMBDA_REC"] * L["rec"], aux=cfg["AUX_CE_WEIGHT"] * L["aux"])
        names = [n for n, p in student.named_parameters() if p.requires_grad]; params = [p for _, p in student.named_parameters() if p.requires_grad]
        groups = ["backbone", "generative", "forensic", "temporal", "fusion", "heads"]; tot = max(1e-12, sum(float(t) for t in terms.values())); SC = 1024.0
        print(f"  {'term':5s} {'weighted':>9s} {'share':>6s} | grad L2-norm per module group (fp16 autocast, loss scaled x{SC:.0f} internally)\n  {'':5s} {'':>9s} {'':>6s} | " + "  ".join(f"{g:>10s}" for g in groups), flush=True)
        for nm, t in terms.items():
            if not t.requires_grad: print(f"  {nm:5s} {float(t):9.4f} {100 * float(t) / tot:5.1f}% | (no graph: term disabled or zero)", flush=True); continue
            gr = torch.autograd.grad(t * SC, params, retain_graph=True, allow_unused=True); sq = defaultdict(float)
            for n_, g_ in zip(names, gr):
                if g_ is not None: sq[param_group_name(n_)] += float(g_.float().pow(2).sum())
            print(f"  {nm:5s} {float(t):9.4f} {100 * float(t) / tot:5.1f}% | " + "  ".join(f"{math.sqrt(sq[g]) / SC:10.3g}" for g in groups), flush=True)
        g = out["gate"].float().mean((0, 1)); act = [GATE_IDX[n] for n in student.streams]; ga = g[act] / g[act].sum()
        print(f"  fusion gate (active streams {student.streams}): {[round(float(x), 3) for x in ga]} | normalised entropy={float(-(ga * ga.clamp_min(1e-7).log()).sum() / math.log(max(2, len(act)))):.3f} (1 = uniform)", flush=True)
        print(f"  KD coverage in this batch (weighting='{cfg.get('KD_WEIGHTING', 'correct_only')}'): {float(L['kd_cov']):.3f}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); log(f"!! DIAGNOSTIC FAILED (training continues): {type(e).__name__}: {e}")
    finally:
        student.zero_grad(set_to_none=True); wrapper.train(was); set_rng_state(rs); torch.cuda.empty_cache()

def nonfinite_grads(model):
    return [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]

def eval_coverage_report(ds, cfg):
    """Evaluation-sampling diagnostic for the longest / median / shortest video: sampled indices, first/last frame, covered fraction. Fails if coverage is wrong."""
    ns = [v["n"] for v in ds.v]; order = sorted(range(len(ns)), key=lambda k: ns[k]); picks = list(dict.fromkeys([order[-1], order[len(order) // 2], order[0]]))
    by_vid = defaultdict(list)
    for it in ds.items: by_vid[it[0]].append(it)
    for vi in picks:
        n = ns[vi]; allidx = []; lines = []
        for _, s, W in by_vid[vi]:
            idx = ds.indices(vi, s, W, None); assert all(b >= a for a, b in zip(idx, idx[1:])), "eval indices not chronological"
            allidx += idx; lines.append(f"      start={s} window={W}: {idx}")
        u = sorted(set(allidx))
        log(f"Eval coverage: '{ds.v[vi]['video_id']}' N={n} clips={len(by_vid[vi])} | first sampled={u[0]} last sampled={u[-1]} | span covered={(u[-1]-u[0]+1)/n:.2f} | distinct frames used={len(u)}/{n}")
        for l in lines: print(l, flush=True)
        if cfg.get("EVAL_CLIP_MODE", "cover") == "cover" and len(by_vid[vi]) >= 2:
            assert u[0] == 0 and u[-1] == n - 1, f"cover mode must reach the first and last frame (got {u[0]}..{u[-1]} of {n})"

# ================= MODIFIED =================
def temporal_demo(train_ds, val_ds, cfg):
    """Preflight demonstration of TRUE temporal windowing (train: one random window; eval: several evenly spaced windows) + determinism."""
    # ---- stronger TRAIN sampling check: longest, median-length and shortest (>= NUM_FRAMES) training videos; window / stride / span / strict order / determinism ----
    T = cfg["NUM_FRAMES"]; cfgW = cfg["TRAIN_WINDOW"]; stride = train_ds.stride; ns = [v["n"] for v in train_ds.v]; order = sorted(range(len(ns)), key=lambda k: ns[k])
    cands = [order[-1], order[len(order) // 2]] + [k for k in order if ns[k] >= T][:1]; cands = list(dict.fromkeys(cands)); old_ep = train_ds.current_epoch; span = (T - 1) * stride + 1
    for vt in cands:
        n = ns[vt]; We = min(cfgW, n)
        log(f"Temporal sampling check (TRAIN): video '{train_ds.v[vt]['video_id']}' | video frame count = {n} | training window = {We} (TRAIN_WINDOW={cfgW}) | frame stride = {stride} | frames/clip = {T}")
        for ep, pos in ((0, 0), (0, 1), (5, 123)):
            train_ds.set_epoch(ep); a_, info = train_ds.indices_ex(vt, None, None, train_ds.clip_rng(pos, vt))
            ds2 = ClipDataset(train_ds.v, cfg, True); ds2.set_epoch(ep); b_, info2 = ds2.indices_ex(vt, None, None, ds2.clip_rng(pos, vt))     # FRESH object, same (epoch, draw position, item)
            assert a_ == b_ and info == info2, f"training clip sampling is NOT deterministic (epoch {ep}, draw {pos}, item {vt})"
            w0 = info["w0"]; assert len(a_) == T and all(0 <= i < n for i in a_), "train indices outside the video"
            if n >= T:
                assert all(y > x for x, y in zip(a_, a_[1:])), f"train indices not strictly increasing: {a_}"
                assert w0 <= min(a_) and max(a_) <= w0 + We - 1, f"train indices {a_} fall outside the chosen window [{w0}, {w0 + We - 1}]"
                assert max(a_) - min(a_) + 1 <= cfgW, f"temporal span {max(a_) - min(a_) + 1} exceeds the configured window {cfgW}"
                if span <= We: assert all(y - x == stride for x, y in zip(a_, a_[1:])), f"frame stride is not {stride}: {a_}"
            else: assert all(y >= x for x, y in zip(a_, a_[1:])), "train indices not chronological (video shorter than NUM_FRAMES -> unavoidable repetition)"
            log(f"   epoch {ep} draw {pos}: window=[{w0},{w0 + We - 1}] sampled frame indices = {a_} | span={max(a_) - min(a_) + 1} -> strictly increasing, inside window, span<=window, deterministic OK")
    i0 = min(123, len(train_ds) - 1); train_ds.set_epoch(5); xa = train_ds[(123, i0)][0]; xb = train_ds[(123, i0)][0]
    assert torch.equal(xa, xb), "epoch 5 + item: pixel output (crop/flip/blur/JPEG) is NOT deterministic"
    log(f"   epoch 5 + item {i0}: identical pixels on re-evaluation (crop/flip/blur/JPEG deterministic)"); train_ds.set_epoch(old_ep)
    eval_coverage_report(val_ds, cfg)
    log("   eval: clip probabilities of all windows are aggregated into ONE video probability (VIDEO_AGGREGATION).")

# ================= MODIFIED =================
def reconstruction_sanity(student, x01, cfg, nmax=4):
    """NUMERICAL sanity of the generative branches on one small batch (finite, in [0,1], same shape) + the warm-up mixing path. NOT evidence that the (randomly initialised) branches are useful."""
    gen = student.gen; xf = x01.flatten(0, 1)[:nmax].contiguous(); was = student.training; student.eval()
    if len(gen.branches) == 0: student.train(was); log("  reconstruction sanity skipped: GENCONVIT_VARIANT='none' (no generative branch)"); return
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=student.amp):
        for br in gen.branches:
            rec, kl = br(xf); rec = rec.float(); nm = "VAE" if br.vae else "AE"
            assert rec.shape == xf.shape, f"{nm} reconstruction shape {tuple(rec.shape)} != input {tuple(xf.shape)}"
            assert torch.isfinite(rec).all() and torch.isfinite(kl.float()).all(), f"{nm} reconstruction / KL contains NaN/Inf"
            assert float(rec.min()) >= -1e-4 and float(rec.max()) <= 1.0 + 1e-4, f"{nm} reconstruction outside [0,1]: [{float(rec.min()):.4f}, {float(rec.max()):.4f}]"
            log(f"  {nm} reconstruction MSE = {float(((rec - xf.float()) ** 2).mean()):.6f}   (range [{float(rec.min()):.3f}, {float(rec.max()):.3f}], finite)")
        fe = {al: gen(xf, gen_alpha=al)[0].float() for al in (0.0, 0.5, 1.0)}
    assert all(torch.isfinite(v).all() for v in fe.values()), "non-finite GenConViT features in the generative-input warm-up path"
    if gen.feed == "recon": assert not torch.allclose(fe[0.0], fe[1.0]), "gen_alpha has no effect on the backbone input"
    student.train(was)
    log("  reconstruction sanity OK (numerical check only: MSE of randomly initialised branches does NOT mean they are already useful; they are learned during training); warm-up mixing path (alpha 0/0.5/1) finite")

def pretrain_generative(cfg, student, train_ds, dev, R):
    """GenConViT pretrains its AE/VAE; here: a short reconstruction-only stage (backbones/other branches untouched). Cached in gen_pretrained.pth."""
    path = os.path.join(R["ckpt"], "gen_pretrained.pth"); gen = student.gen
    if os.path.exists(path): gen.branches.load_state_dict(load_ckpt(path)); log(f"Loaded pretrained generative branches from {path}"); return
    steps, bsz = cfg["GEN_PRETRAIN_STEPS"], cfg["GEN_PRETRAIN_BATCH"]
    opt = torch.optim.AdamW(gen.branches.parameters(), lr=cfg["GEN_PRETRAIN_LR"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    scaler = make_scaler(cfg["USE_AMP"]); train_ds.set_epoch(10_000)
    saved_cfg = train_ds.cfg
    if cfg.get("GEN_PRETRAIN_CLEAN", True): train_ds.cfg = dict(cfg, AUG_BLUR_PROB=0.0, AUG_JPEG_PROB=0.0, AUG_DOWNSCALE_PROB=0.0, AUG_COLOR_PROB=0.0, AUG_NOISE_PROB=0.0, SBI_PROB=0.0)   # reconstruct CLEAN frames
    pool = np.array([i for i, (vi, _, _) in enumerate(train_ds.items) if (not cfg.get("GEN_PRETRAIN_REAL_ONLY", False)) or train_ds.v[vi]["label"] == 0])
    g = torch.Generator().manual_seed(cfg["SEED"] + 777); keys = list(enumerate(pool[torch.randint(0, len(pool), (steps * bsz,), generator=g).numpy()].tolist()))
    dl = DataLoader(train_ds, batch_size=bsz, sampler=keys, num_workers=cfg["NUM_WORKERS"], drop_last=True, prefetch_factor=4 if cfg["NUM_WORKERS"] > 0 else None)
    gen.branches.train(); log(f"Stage 0: pretraining generative branches (reconstruction only): {steps} steps x {bsz} clips x {cfg['NUM_FRAMES']} frames")
    run = None
    for k, (x, _, _) in enumerate(tqdm(dl, total=steps, desc="gen-pretrain", leave=False)):
        xf = to_x01(x, dev).flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["USE_AMP"]):
            loss = 0.0; mse = 0.0
            for br in gen.branches:
                rec, kl = br(xf); m_ = ((rec.float() - xf.float()) ** 2).mean(); mse = mse + m_.detach(); loss = loss + m_ + cfg["LAMBDA_KL"] * kl.float().mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(gen.branches.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        run = float(mse) / len(gen.branches) if run is None else 0.98 * run + 0.02 * float(mse) / len(gen.branches)
        if (k + 1) % 100 == 0: log(f"  gen-pretrain step {k+1}/{steps} | mean branch MSE (EMA) = {run:.5f}")
    train_ds.cfg = saved_cfg; del dl; gc.collect(); atomic_save(gen.branches.state_dict(), path)
    log(f"Stage 0 done: final mean branch MSE ~ {run:.5f}" + ("   !! still > 0.03: reconstructions are poor, raise GEN_PRETRAIN_STEPS" if run > 0.03 else ""))
    
def preflight(cfg, R, splits, student, teacher, wrapper_raw, train_ds, val_ds, dev):
    log("=================== PREFLIGHT / SANITY CHECKS ===================")
    snap = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}; rs = get_rng_state()
    x0, y0, _ = train_ds[0]; v0 = train_ds.v[train_ds.items[0][0]]
    log(f"Sample[0]: tensor {tuple(x0.shape)} dtype={x0.dtype} label={int(y0)} ({v0['generator']}; {v0['video_id']}) frames={v0['n']}")
    assert x0.shape == (cfg["NUM_FRAMES"], cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"], 3)
    temporal_demo(train_ds, val_ds, cfg)
    for lab in (0, 1):
        ex = next(v for v in splits["train"] if v["label"] == lab); log(f"  label {lab} <- folder '{ex['generator']}' (e.g. {ex['vname']})")
    for s in splits: c = np.bincount([v["label"] for v in splits[s]], minlength=2); log(f"  {s}: real={c[0]} fake={c[1]} videos")
    bs0 = 2; xb = torch.stack([train_ds[j][0] for j in range(bs0)]); yb = torch.tensor([int(train_ds[j][1]) for j in range(bs0)])
    x01 = to_x01(xb, dev); y = yb.to(dev); wrapper_raw.train()
    log("Tensor-shape dry-run (one batch):"); out = wrapper_raw.student(x01, debug=True)
    B, T = bs0, cfg["NUM_FRAMES"]; assert tuple(out["frame_logits"].shape) == (B, T, 2) and tuple(out["video_logits"].shape) == (B, 2)
    reconstruction_sanity(student, x01, cfg)
    out = wrapper_raw(x01, gen_alpha=0.5)          # dry-run forward/backward through the generative-input MIXING path
    if teacher is not None:
        tl = out["t_logits"]; assert tuple(tl.shape) == (B, T, 2)
        log(f"  Teacher frame logits [B,T,2]     = {list(tl.shape)}\n  Teacher video distrib [B,2]      = {list(F.softmax(tl / cfg['TEMPERATURE'], -1).mean(1).shape)}")
    L = compute_loss(out, y, cfg); bad = nonfinite_report(out, L); assert not bad, f"NaN/Inf in {bad}"
    L["total"].backward(); gb = nonfinite_grads(student); assert not gb, f"non-finite grads: {gb[:5]}"
    log(f"  forward+backward OK | loss total={L['total'].item():.4f} ce={L['ce'].item():.4f} kd={float(L['kd'].detach()):.4f} tc={float(L['tc'].detach()):.4f} rec={L['rec'].item():.4f}")
    if cfg["USE_FREQUENCY"]: fa = out["freq_absmax"]; assert torch.isfinite(fa).all() and (fa > 0).all(), "frequency branch dead"; log(f"  frequency branch alive (|f|max={fa.max().item():.3f})")
    if cfg["USE_SPATIAL_ARTIFACT"]: aa = out["art_absmax"]; assert torch.isfinite(aa).all() and (aa > 0).all(), "spatial-artifact branch dead"; log(f"  spatial-artifact branch alive (|a|max={aa.max().item():.3f})")
    if len(student.streams) > 1: log(f"  fusion gates (spatial, artifact, frequency) mean = {[round(float(x), 3) for x in out['gate'].float().mean((0, 1))]} (sum over active = 1)")
    if cfg["USE_TEMPORAL"]:
        ex = student(x01, extras=True); assert ex["ex_attn"].shape[-1] == T + 1; log(f"  temporal branch alive (attention {tuple(ex['ex_attn'].shape)})")
    if cfg["USE_KD"]: assert torch.isfinite(L["kd"]) and L["kd"].item() >= 0; log(f"  KD loss finite ({L['kd'].item():.4f}); any teacher param requires_grad: {any(p.requires_grad for p in teacher.parameters())}")
    student.zero_grad(set_to_none=True)
    tmp = os.path.join(R["ckpt"], "_roundtrip.pth"); atomic_save({"model": student.state_dict()}, tmp); back = load_ckpt(tmp)["model"]; os.remove(tmp)
    assert all(torch.equal(back[k].cpu(), v.cpu()) for k, v in student.state_dict().items() if v.is_floating_point()), "checkpoint round-trip mismatch"; log("  checkpoint save/load round-trip OK")
    bs = cfg["BATCH_SIZE"]; ng = torch.cuda.device_count() if cfg["USE_DATAPARALLEL"] else 1
    cands = [bs] if bs != "auto" else [c for c in (8, 4, 2) if c <= cfg["TARGET_EFFECTIVE_BATCH"] and c % ng == 0] or [ng]
    chosen = None; dp = make_parallel(wrapper_raw, cfg)
    for c in cands:
        try:
            torch.cuda.empty_cache(); [torch.cuda.reset_peak_memory_stats(i) for i in range(torch.cuda.device_count())]
            xp = torch.rand(c, cfg["NUM_FRAMES"], 3, cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"], device=dev); yp = torch.randint(0, 2, (c,), device=dev)
            for p_ in student.gen.backbones.parameters(): p_.requires_grad_(True)
            o = dp(xp); compute_loss(o, yp, cfg)["total"].backward(); student.zero_grad(set_to_none=True); chosen = c
            log(f"  batch probe: {c} clips/step fits ({c//ng} per GPU) | peak mem {gpu_mem()}"); break
        except torch.cuda.OutOfMemoryError:
            student.zero_grad(set_to_none=True); torch.cuda.empty_cache(); log(f"  batch probe: {c} clips/step -> OOM")
    if chosen is None: raise RuntimeError("OOM even at the smallest batch. Set GRAD_CHECKPOINT=True or GENCONVIT_VARIANT='ed'.")
    student.load_state_dict({k: v.to(dev) for k, v in snap.items()}); set_rng_state(rs); student.zero_grad(set_to_none=True)
    log("  (model forward/backward preflight passed; checkpoint/resume self-test follows once optimizer & scheduler exist)")
    return chosen

# =====================================================================================
#                                      TRAINING
# =====================================================================================
def finite_vals(hist, key): return [r[key] for r in hist if math.isfinite(r.get(key, float("nan")))]

def trend_message(hist):
    h = [r for r in hist if math.isfinite(r.get("val_auc", float("nan")))][-4:]
    if len(h) < 2: return "Not enough validated epochs yet to describe a trend."
    da = h[-1]["val_auc"] - h[0]["val_auc"]; dl = h[-1]["val_loss"] - h[0]["val_loss"]
    if da > 0.002: return "Validation AUC has improved over the latest epochs."
    if da < -0.002 and dl > 0: return "Validation loss is increasing while AUC is decreasing."
    if abs(da) <= 0.002: return "Validation AUC has plateaued."
    return "Validation AUC has decreased slightly over the latest epochs."

def est_val_minutes(S, cfg):
    t = finite_vals(S.hist, "val_time_min")[-3:]
    return float(np.mean(t)) if t else cfg["VAL_TIME_GUESS_MINUTES"]

# ================= MODIFIED =================
def prerun_summary(cfg, splits, student, teacher, tval, cost, rpath, R, tag):
    on = lambda b: "ON" if b else "OFF"; nfr = lambda s: sum(v["n"] for v in splits[s])
    best = os.path.join(R["ckpt"], "best_auc.pth"); est = est_val_minutes(SimpleNamespace(hist=[]), cfg)
    print("\n" + "=" * 78 + "\nPRE-RUN SUMMARY")
    print(f"DATA:\n  FF++ train = {len(splits['train'])} videos ({nfr('train')} frames)\n  FF++ val   = {len(splits['val'])} videos ({nfr('val')} frames)\n  FF++ test  = {len(splits['test'])} videos ({nfr('test')} frames)")
    print(f"  grouping modes: FF++={cfg['FFPP_GROUPING_MODE']} CelebDF={cfg['CELEBDF_GROUPING_MODE']} DFDC={cfg['DFDC_GROUPING_MODE']}")
    print(f"MODEL:\n  GenConViT-based = ON ({GENCONVIT_NOTE}; variant '{cfg['GENCONVIT_VARIANT']}')\n  Spatial Artifact = {on(cfg['USE_SPATIAL_ARTIFACT'])}\n  Frequency = {on(cfg['USE_FREQUENCY'])}\n"
          f"  Temporal = {on(cfg['USE_TEMPORAL'])}\n  KD = {on(cfg['USE_KD'])}\n  EMA = {on(cfg['USE_EMA'])}\n  model tag = {tag} | generative loss warm-up = {cfg['GENERATIVE_WARMUP_EPOCHS']} epoch(s) | generative-input warm-up = {cfg['GENERATIVE_INPUT_WARMUP_EPOCHS']} epoch(s)")
    print(f"TEMPORAL:\n  frames/clip = {cfg['NUM_FRAMES']}\n  train window = {cfg['TRAIN_WINDOW']}\n  stride = {1 if cfg['TEMPORAL_SAMPLING'] == 'contiguous' else cfg['TEMPORAL_STRIDE']}\n"
          f"  eval window = {cfg['EVAL_WINDOW']} (hop {cfg['EVAL_WINDOW_STRIDE']}, max {cfg['MAX_EVAL_CLIPS']} clips/video)")
    print(f"COMPUTE:\n  parameters = {cost['params']/1e6:.2f} M\n  trainable parameters = {cost['trainable']/1e6:.2f} M\n  estimated GFLOPs/clip = {cost['gflops_clip']:.1f}  (backbone passes/clip = {cost['backbone_passes']})")
    if teacher is not None:
        frozen = all(not p.requires_grad for p in teacher.parameters()) and not teacher.training
        print(f"TEACHER:\n  loaded = YES\n  frozen = {'YES' if frozen else 'NO  <-- PROBLEM'}\n  validation AUC = {('%.4f' % tval['auc']) if tval else 'not evaluated (RUN_TEACHER_PREFLIGHT=False)'}")
    else: print("TEACHER:\n  loaded = N/A (KD is OFF)")
    print(f"CHECKPOINT:\n  resume = {rpath if rpath else 'NO (fresh run)'}\n  best = {best if os.path.exists(best) else '(none yet) ' + best}")
    print(f"SESSION:\n  maximum budget = {cfg['MAX_SESSION_MINUTES']} min\n  safety reserve = {cfg['DIAG_RESERVE_MINUTES']} min (+ estimated validation ~{est:.1f} min)\n" + "=" * 78 + "\n", flush=True)

def run_train(cfg, R, dev):
    log("---- building datasets ----")
    ffpp = build_videos("FF++", cfg["FFPP_PATH"], "ffpp", cfg["FFPP_GROUPING_MODE"], seed=cfg["SEED"], cfg=cfg); describe(ffpp, "FF++ all")
    splits, sig = prerun_guard("FF++ official split mapping / real-video counts", ffpp_split, ffpp, cfg, R)
    def _grouping_checks():
        for sp in ("train", "val", "test"): grouping_check(splits[sp], f"FF++ {sp}", cfg, cfg["FFPP_GROUPING_MODE"])
        ffpp_structure_diagnostic([v for sp in ("train", "val", "test") for v in splits[sp]], cfg)
    prerun_guard("FF++ frame grouping / dataset structure", _grouping_checks)
    shortcut_probe(splits["train"])
    pd.DataFrame([{"video_id": v["video_id"], "split": s, "label": v["label"], "generator": v["generator"], "n_frames": v["n"]} for s in splits for v in splits[s]]).to_csv(os.path.join(R["configs"], "ffpp_split.csv"), index=False)
    train_ds = ClipDataset(splits["train"], cfg, True)
    assert cfg["VAL_MODE"] in ("fast", "full"), "VAL_MODE must be 'fast' or 'full'"
    val_videos = fast_val_subset(splits["val"], cfg["VAL_FAST_REAL"], cfg["VAL_FAST_PER_GROUP"], cfg["SEED"]) if cfg["VAL_MODE"] == "fast" else splits["val"]
    val_ds = ClipDataset(val_videos, cfg, False, cfg["VAL_FAST_CLIPS"] if cfg["VAL_MODE"] == "fast" else cfg["VAL_FULL_CLIPS"])
    log(f"Per-epoch validation: mode={cfg['VAL_MODE']} | {len(val_videos)} val videos (real={sum(v['label']==0 for v in val_videos)}) | {len(val_ds)} clips | EVAL_CLIP_MODE={cfg.get('EVAL_CLIP_MODE','cover')}")
    labels = np.array([splits["train"][vi]["label"] for vi, _, _ in train_ds.items]); nr, nf = (labels == 0).sum(), (labels == 1).sum()
    w_t = torch.tensor(np.where(labels == 0, 0.5 / max(1, nr), 0.5 / max(1, nf)), dtype=torch.double); log(f"Class-balanced sampling: real={nr} fake={nf}")
    if cfg["SAMPLER_MODE"] == "stratified":
        grp = {}
        for i, (vi, _, _) in enumerate(train_ds.items):
            v_ = splits["train"][vi]; grp.setdefault(v_["generator"], (v_["label"], []))[1].append(i)
        w_t = StratifiedPlan(grp, cfg["REAL_FRAC"]); log(f"Stratified sampling: real {cfg['REAL_FRAC']:.0%} + equal quota per manipulation folder {sorted(grp)}")
    student = StudentModel(cfg).to(dev); tag = model_tag(cfg)
    tot, trn = print_model_summary(cfg, student, dev); write_ablation_csv(cfg, R, tag)
    teacher = prerun_guard("teacher loading", load_teacher, cfg, dev) if cfg["USE_KD"] else None
    wrapper = TrainWrapper(student, teacher, cfg).to(dev); tval = None
    if teacher is not None and cfg["RUN_TEACHER_PREFLIGHT"]:
        mv, mf = eval_teacher(teacher, val_ds, cfg); tval = mv; pd.DataFrame([dict(level="video", **mv), dict(level="frame", **mf)]).to_csv(os.path.join(R["metrics"], "teacher_validation_metrics.csv"), index=False)
        log(f"Teacher FF++ val: video AUC={mv['auc']:.4f} F1={mv['f1']:.4f} | frame AUC={mf['auc']:.4f}")
        if mv["auc"] < 0.5: raise RuntimeError("Teacher val AUC < 0.5 -> label order is inverted. Set TEACHER_FAKE_INDEX=0 (or fix the checkpoint).")
    elif teacher is not None: log("RUN_TEACHER_PREFLIGHT=False: skipping teacher validation pass (label order NOT re-verified in this run).")
    cost = cost_summary(cfg, student, teacher, dev)
    rpath = cfg["RESUME_CHECKPOINT"] or (os.path.join(R["ckpt"], "last.pth") if cfg["AUTO_RESUME"] and os.path.exists(os.path.join(R["ckpt"], "last.pth")) else "")
    if rpath and not os.path.exists(rpath): raise FileNotFoundError(f"RESUME_CHECKPOINT not found: {rpath}")
    prerun_summary(cfg, splits, student, teacher, tval, cost, rpath, R, tag)
    bs = prerun_guard("preflight (temporal sampling / reconstruction / shapes / KD / branches / checkpoint round-trip)", preflight, cfg, R, splits, student, teacher, wrapper, train_ds, val_ds, dev) if cfg["RUN_PREFLIGHT"] else (cfg["BATCH_SIZE"] if cfg["BATCH_SIZE"] != "auto" else 4)
    accum = max(1, math.ceil(cfg["TARGET_EFFECTIVE_BATCH"] / bs)); n_samples = cfg["TRAIN_CLIPS_PER_EPOCH"] or len(train_ds)
    steps_per_epoch = max(1, (n_samples // bs) // accum); total_steps = steps_per_epoch * cfg["NUM_EPOCHS"]; warm = max(1, cfg["WARMUP_EPOCHS"] * steps_per_epoch)
    log(f"Batch: {bs} clips/step x accum {accum} = effective {bs*accum} | steps/epoch={steps_per_epoch} | total steps={total_steps}")
    opt = make_optimizer(student, cfg)
    lam = lambda s: (s + 1) / warm if s < warm else cfg["MIN_LR_RATIO"] + (1 - cfg["MIN_LR_RATIO"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam); scaler = make_scaler(cfg["USE_AMP"]); ema = EMA(student, cfg["EMA_DECAY"]) if cfg["USE_EMA"] else None
    S = SimpleNamespace(cfg=cfg, student=student, opt=opt, sched=sched, scaler=scaler, ema=ema, epochs_done=0, global_step=0, hist=[], sig=sig, finished=False, partial=None, steps_per_epoch=steps_per_epoch,
                        best=dict(auc=-1.0, f1=-1.0, bal_acc=-1.0, epoch=-1, source="raw"))
    if cfg["RUN_PREFLIGHT"]:
        prerun_guard("checkpoint/resume self-test", resume_selftest, cfg, R, S, steps_per_epoch, accum, bs, w_t, n_samples); log("=================== PREFLIGHT PASSED ===================")
    if cfg.get("RUN_LOSS_DIAGNOSTIC", True): loss_gradient_diagnostic(wrapper, student, train_ds, cfg, dev)
    if not rpath and cfg["GEN_PRETRAIN_STEPS"] > 0 and len(student.gen.branches) > 0:
        pretrain_generative(cfg, student, train_ds, dev, R)
        if ema is not None: ema.m.load_state_dict(student.state_dict())     # EMA was deep-copied BEFORE pretraining: re-sync it
        set_seed(cfg["SEED"])
    start_step_in_epoch = 0; resume_acc = None
    if rpath:
        ck = load_ckpt(rpath); check_arch(ck["config"], cfg); check_gen_input(ck, cfg, steps_per_epoch)
        for k in ("NUM_EPOCHS", "TARGET_EFFECTIVE_BATCH", "LR_BACKBONE", "LR_NEW_MODULES", "LAMBDA_KD", "LAMBDA_REC", "LAMBDA_KL", "GENERATIVE_WARMUP_EPOCHS", "GENERATIVE_INPUT_WARMUP_EPOCHS", "TRAIN_CLIPS_PER_EPOCH", "SEED", "TRAIN_WINDOW", "TEMPORAL_STRIDE"):
            if ck["config"].get(k) != cfg[k]: log(f"WARN: {k} differs from checkpoint ({ck['config'].get(k)} -> {cfg[k]}); schedule/sampling may not continue identically.")
        if ck.get("split_signature") != sig: raise RuntimeError("Validation/test split signature differs from the checkpoint run — dataset paths/splits changed; refusing to resume.")
        plan = plan_resume(ck, steps_per_epoch, accum, bs)
        student.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema"): ema.load_state_dict(ck["ema"])
        S.epochs_done, S.global_step, S.hist, S.best, S.finished = ck["epoch"], ck["global_step"], ck["history"], ck["best"], ck.get("finished", False)
        start_step_in_epoch = plan["start_step"]; resume_acc = plan["acc"]
        set_rng_state(ck["rng"])
        log(f"RESUMED from {rpath}: epochs_done={S.epochs_done}, global_step={S.global_step}, best AUC={S.best['auc']:.4f}, "
            f"{'mid-epoch: will continue at update step %d/%d' % (plan['next_update_step'], steps_per_epoch) if start_step_in_epoch else 'epoch boundary'}")
        if S.finished: log("This run was already marked finished (all epochs / early stop). Use MODE='evaluate'."); return
    json.dump(dict(config=cfg_to_jsonable(cfg), model_tag=tag, genconvit_note=GENCONVIT_NOTE, genconvit_detail=GENCONVIT_DETAIL, generative_input_warmup_epochs=cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"], env=R["env"], batch_size=bs, grad_accum=accum, effective_batch=bs * accum, optimizer="AdamW", scheduler="warmup+cosine (per step)",
                   n_train_videos=len(splits["train"]), n_val_videos=len(splits["val"]), n_test_videos=len(splits["test"]), params_total=tot, params_trainable_initial=trn,
                   param_breakdown={n: t for n, t, _ in param_breakdown(student)}, teacher=cfg["TEACHER_CHECKPOINT"] if cfg["USE_KD"] else None, split_signature=sig),
              open(os.path.join(R["configs"], "experiment_config.json"), "w"), indent=2, default=str)
    dpw = make_parallel(wrapper, cfg); dp_student = make_parallel(student, cfg)
    stop_reason = None; bad_consec = 0; bad_grad = 0; nan_ckpts = 0; last_emerg = time.time(); epochs_this_session = 0; ok_steps = 0; skipped_steps = 0
    log(f"================ TRAINING START | epochs {S.epochs_done+1}..{cfg['NUM_EPOCHS']} | budget {cfg['MAX_SESSION_MINUTES']} min (reserve {cfg['DIAG_RESERVE_MINUTES']}) ================")
    for epoch in range(S.epochs_done, cfg["NUM_EPOCHS"]):
        if cfg["MAX_EPOCHS_THIS_SESSION"] and epochs_this_session >= cfg["MAX_EPOCHS_THIS_SESSION"]: stop_reason = "MAX_EPOCHS_THIS_SESSION reached"; break
        est_val = est_val_minutes(S, cfg); deadline = cfg["MAX_SESSION_MINUTES"] - cfg["DIAG_RESERVE_MINUTES"] - est_val
        if elapsed_min() >= deadline: stop_reason = f"session time budget reached before starting epoch {epoch+1}"; break
        t_ep = time.time(); frozen = epoch < cfg["FREEZE_BACKBONE_EPOCHS"]; S.partial = None
        student.gen.backbones.requires_grad_(not frozen); dpw.train()
        gs0 = gen_scale(cfg, S.global_step, steps_per_epoch)
        log(f"Epoch {epoch+1}: effective generative weights at start: L_REC x {cfg['LAMBDA_REC']*gs0:.4f} | L_KL x {cfg['LAMBDA_KL']*gs0:.6f} (warm-up factor {gs0:.3f})")
        ga0 = gen_input_alpha(cfg, S.global_step, steps_per_epoch); ga1_exp = gen_input_alpha(cfg, (epoch + 1) * steps_per_epoch, steps_per_epoch)
        log(f"Epoch {epoch+1}: generative INPUT alpha (reconstruction fraction in backbone input) = {ga0:.3f} -> {ga1_exp:.3f} (expected at epoch end; global_step {S.global_step}, steps/epoch {steps_per_epoch})")
        # deterministic epoch order + deterministic per-item clip/augmentation RNG (seed, epoch, draw position, item)
        train_ds.set_epoch(epoch)
        keys = epoch_keys(w_t, n_samples, cfg["SEED"], epoch); sampler_report(keys, train_ds, splits["train"], epoch)
        full_micro = steps_per_epoch * accum; skip_micro = start_step_in_epoch * accum
        sub = keys[skip_micro * bs: full_micro * bs]
        dgen = torch.Generator().manual_seed(cfg["SEED"] * 1000 + epoch * 10 + (1 if skip_micro else 0))
        dl = DataLoader(train_ds, batch_size=bs, sampler=sub, num_workers=cfg["NUM_WORKERS"], pin_memory=False, drop_last=True, generator=dgen, prefetch_factor=4 if cfg["NUM_WORKERS"] > 0 else None)
        total_micro = len(sub) // bs; step_in_epoch = start_step_in_epoch
        acc = {k: torch.zeros((), device=dev) for k in ("loss", "ce", "kd", "tc", "rec", "correct", "n", "aux0", "aux1", "aux2")}
        if skip_micro:
            log(f"Resuming epoch {epoch+1} at update step {start_step_in_epoch+1}/{steps_per_epoch}: skipping {skip_micro} already-processed micro-batches (draw positions < {skip_micro*bs})")
            if resume_acc:
                for k in acc: acc[k] += float(resume_acc.get(k, 0.0))
        start_step_in_epoch = 0; resume_acc = None
        pbar = tqdm(total=total_micro, desc=f"Epoch {epoch+1}/{cfg['NUM_EPOCHS']}{' [backbone frozen]' if frozen else ''}", leave=False); opt.zero_grad(set_to_none=True); partial_stop = False; window_bad = False
        acc_dump = lambda: {k: float(v.item()) for k, v in acc.items()}
        dl_it = iter(dl)
        for mb, (x, y, _) in enumerate(dl_it):
            if elapsed_min() >= deadline: partial_stop = True; break
            x01 = to_x01(x, dev); y = y.to(dev, non_blocking=True)
            try:
                ks = kd_scale(cfg, S.global_step, steps_per_epoch)
                out = dpw(x01, gen_alpha=gen_input_alpha(cfg, S.global_step, steps_per_epoch), use_teacher=ks > 0.0); L = compute_loss(out, y, cfg, gen_scale(cfg, S.global_step, steps_per_epoch), ks)
            except torch.cuda.OutOfMemoryError:
                S.epochs_done = epoch; save_full_ckpt(os.path.join(R["ckpt"], "emergency_oom.pth"), S, partial=dict(epoch=epoch, step_in_epoch=step_in_epoch, micro_in_epoch=step_in_epoch * accum, steps_per_epoch=steps_per_epoch, acc=acc_dump()), tag="oom")
                raise RuntimeError(f"CUDA OOM at batch {bs}. Set BATCH_SIZE to {max(1, bs//2)} (or 'auto'), enable GRAD_CHECKPOINT, or use GENCONVIT_VARIANT='ed'. Emergency checkpoint saved (weights unchanged).")
            if not torch.isfinite(L["total"]):
                bad_consec += 1; comp = nonfinite_report(out, L); log(f"!! NaN/Inf loss at global_step {S.global_step}: components {comp} (consecutive {bad_consec}) -> update skipped")
                if nan_ckpts < 3: S.epochs_done = epoch; save_full_ckpt(os.path.join(R["ckpt"], "emergency_nan.pth"), S, partial=dict(epoch=epoch, step_in_epoch=step_in_epoch, micro_in_epoch=step_in_epoch * accum, steps_per_epoch=steps_per_epoch, acc=acc_dump()), tag="nan"); nan_ckpts += 1
                if bad_consec >= cfg["NAN_PATIENCE"]: raise RuntimeError(f"{bad_consec} consecutive non-finite losses (components: {comp}). Last-good weights in emergency_nan.pth.")
                window_bad = True                                   # the whole accumulation window is discarded at its boundary
                if (mb + 1) % accum == 0:
                    opt.zero_grad(set_to_none=True); window_bad = False; skipped_steps += 1; step_in_epoch += 1
                    log(f"!! optimizer step SKIPPED (reason: non-finite loss) | ok={ok_steps} skipped={skipped_steps}")
                pbar.update(1); continue
            try: scaler.scale(L["total"] / accum).backward()
            except torch.cuda.OutOfMemoryError:
                S.epochs_done = epoch; save_full_ckpt(os.path.join(R["ckpt"], "emergency_oom.pth"), S, partial=dict(epoch=epoch, step_in_epoch=step_in_epoch, micro_in_epoch=step_in_epoch * accum, steps_per_epoch=steps_per_epoch, acc=acc_dump()), tag="oom")
                raise RuntimeError(f"CUDA OOM in backward at batch {bs}. Set BATCH_SIZE to {max(1, bs//2)}, enable GRAD_CHECKPOINT, or use GENCONVIT_VARIANT='ed'. Emergency checkpoint saved.")
            bad_consec = 0
            with torch.no_grad():
                B_ = y.shape[0]; acc["loss"] += L["total"].detach() * B_; acc["ce"] += L["ce"].detach() * B_; acc["kd"] += L["kd"].detach() * B_; acc["tc"] += L["tc"].detach() * B_
                acc["rec"] += L["rec"].detach() * B_; acc["correct"] += (out["video_logits"].argmax(1) == y).sum(); acc["n"] += B_
                ah_ = (out["aux_logits"].detach().argmax(-1) == y.view(-1, 1, 1)).float().mean((0, 1))      # per-branch frame accuracy [S]
                for k_ in range(ah_.shape[0]): acc[f"aux{k_}"] += ah_[k_] * B_
            if (mb + 1) % accum == 0:
                if window_bad:                                      # a micro-batch of this window had a non-finite loss -> discard the window
                    skipped_steps += 1; window_bad = False
                    log(f"!! optimizer step SKIPPED (reason: non-finite loss earlier in this accumulation window) | ok={ok_steps} skipped={skipped_steps}")
                else:
                    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["GRAD_CLIP"])
                    if torch.isfinite(gn):                          # REAL optimizer update: scheduler, EMA and global_step advance ONLY here
                        bad_grad = 0; scaler.step(opt); scaler.update(); sched.step()
                        if ema is not None: ema.update(student)
                        S.global_step += 1; ok_steps += 1
                    else:                                           # skipped: no opt.step / sched.step / ema.update / global_step; scaler backs off its scale
                        bad_grad += 1; skipped_steps += 1; scaler.update()
                        log(f"!! optimizer step SKIPPED (reason: non-finite grad norm) | ok={ok_steps} skipped={skipped_steps} consecutive={bad_grad} | scaler scale={scaler.get_scale() if scaler.is_enabled() else 1.0:.0f}")
                opt.zero_grad(set_to_none=True); step_in_epoch += 1  # step_in_epoch = position in the epoch's SAMPLE PLAN (consumed either way) -> resume stays exact
                if bad_grad >= 30: save_full_ckpt(os.path.join(R["ckpt"], "emergency_nan.pth"), S, partial=dict(epoch=epoch, step_in_epoch=step_in_epoch, micro_in_epoch=step_in_epoch * accum, steps_per_epoch=steps_per_epoch, acc=acc_dump()), tag="nan"); raise RuntimeError("Persistent non-finite gradients; stopping.")
                if step_in_epoch % 10 == 0:
                    n_ = max(1.0, acc["n"].item()); pbar.set_postfix(loss=f"{acc['loss'].item()/n_:.4f}", lr=f"{opt.param_groups[-1]['lr']:.2e}", t=f"{elapsed_min():.0f}m", gpu=gpu_mem())
                if (time.time() - last_emerg) / 60 >= cfg["EMERGENCY_CKPT_MINUTES"]:
                    S.epochs_done = epoch; S.partial = dict(epoch=epoch, step_in_epoch=step_in_epoch, micro_in_epoch=step_in_epoch * accum, steps_per_epoch=steps_per_epoch, acc=acc_dump())
                    save_full_ckpt(os.path.join(R["ckpt"], "last.pth"), S, partial=S.partial, tag="periodic"); last_emerg = time.time(); log(f"periodic mid-epoch checkpoint saved (epoch {epoch+1}, step {step_in_epoch}/{steps_per_epoch})")
            pbar.update(1)
        pbar.close(); close_iter(dl_it); del dl_it, dl
        if partial_stop:
            S.epochs_done = epoch; S.partial = dict(epoch=epoch, step_in_epoch=step_in_epoch, micro_in_epoch=step_in_epoch * accum, steps_per_epoch=steps_per_epoch, acc=acc_dump())
            stop_reason = f"time budget hit mid-epoch {epoch+1} (update step {step_in_epoch}/{steps_per_epoch})"; break
        # ------------------------ end of epoch ------------------------
        n_ = max(1.0, acc["n"].item()); tr = {k: acc[k].item() / n_ for k in ("loss", "ce", "kd", "tc", "rec")}; tr_acc = acc["correct"].item() / n_
        log(f"Epoch {epoch+1}: TRAIN frame-accuracy per branch head: " + ", ".join(f"{nm}={acc[f'aux{k}'].item() / n_:.3f}" for k, nm in enumerate(student.streams)))
        S.epochs_done = epoch + 1; S.partial = None; epochs_this_session += 1
        gs1 = gen_scale(cfg, S.global_step, steps_per_epoch); w_rec_end, w_kl_end = cfg["LAMBDA_REC"] * gs1, cfg["LAMBDA_KL"] * gs1
        log(f"Epoch {epoch+1}: effective generative weights at end: L_REC x {w_rec_end:.4f} | L_KL x {w_kl_end:.6f} (warm-up factor {gs1:.3f})")
        ga_end = gen_input_alpha(cfg, S.global_step, steps_per_epoch); log(f"Epoch {epoch+1}: generative INPUT alpha: {ga0:.3f} -> {ga_end:.3f}")
        log(f"Epoch {epoch+1}: optimizer steps OK={ok_steps} skipped={skipped_steps} (cumulative, this session) | global_step={S.global_step} | GradScaler scale={scaler.get_scale() if scaler.is_enabled() else 1.0:.0f}")
        remaining = cfg["MAX_SESSION_MINUTES"] - elapsed_min(); est_val = est_val_minutes(S, cfg)
        nan_ = float("nan"); base_row = dict(epoch=epoch + 1, train_loss=tr["loss"], train_acc=tr_acc, train_ce=tr["ce"], train_kd=tr["kd"], train_tc=tr["tc"], train_rec=tr["rec"],
                                             w_rec_eff=w_rec_end, w_kl_eff=w_kl_end, gen_warmup_factor=gs1, gen_input_alpha_start=ga0, gen_input_alpha=ga_end,
                                             lr_backbone=opt.param_groups[0]["lr"], lr_new=opt.param_groups[-1]["lr"], epoch_time_min=(time.time() - t_ep) / 60, global_step=S.global_step)
        if remaining < est_val + cfg["DIAG_RESERVE_MINUTES"]:
            log(f"Only {remaining:.1f} min left (< est. validation {est_val:.1f} + reserve {cfg['DIAG_RESERVE_MINUTES']}) -> SKIPPING validation, saving checkpoint and stopping.")
            S.hist.append(dict(base_row, val_loss=nan_, val_acc=nan_, val_bal_acc=nan_, val_precision=nan_, val_recall=nan_, val_specificity=nan_, val_f1=nan_, val_mcc=nan_, val_auc=nan_, val_pr_auc=nan_,
                               weights_used="", raw_val_auc=nan_, ema_val_auc=nan_, val_time_min=nan_, val_skipped=True, elapsed_min=elapsed_min()))
            pd.DataFrame(S.hist).to_csv(os.path.join(R["metrics"], "training_history.csv"), index=False)
            stop_reason = "time budget: epoch-end validation skipped"; break
        if remaining < 3 * est_val + cfg["DIAG_RESERVE_MINUTES"] + 10:          # tight: persist the trained epoch BEFORE validating
            save_full_ckpt(os.path.join(R["ckpt"], "last.pth"), S, tag="epoch_end_pre_validation"); log("time is tight: pre-validation checkpoint saved")
        t_v = time.time(); vm = {"raw": validate(dp_student, val_ds, cfg)}
        if ema is not None: vm["ema"] = validate(make_parallel(ema.m, cfg), val_ds, cfg)
        val_time = (time.time() - t_v) / 60
        src = max(vm, key=lambda k: vm[k]["auc"] if not math.isnan(vm[k]["auc"]) else -1); v = vm[src]; improved = v["auc"] > S.best["auc"]
        if improved: S.best.update(auc=v["auc"], epoch=epoch + 1, source=src)
        S.best["f1"] = max(S.best["f1"], v["f1"]); S.best["bal_acc"] = max(S.best["bal_acc"], v["bal_acc"])
        row = dict(base_row, val_loss=v["loss"], val_acc=v["acc"], val_bal_acc=v["bal_acc"], val_precision=v["precision"], val_recall=v["recall"], val_specificity=v["specificity"], val_f1=v["f1"],
                   val_mcc=v["mcc"], val_auc=v["auc"], val_pr_auc=v["pr_auc"], val_f1_opt=v["f1_opt"], val_bal_acc_opt=v["bal_acc_opt"], val_thr_opt=v["thr_opt"], weights_used=src, raw_val_auc=vm["raw"]["auc"], ema_val_auc=vm["ema"]["auc"] if "ema" in vm else nan_,
                   val_time_min=val_time, val_skipped=False, elapsed_min=elapsed_min())
        S.hist.append(row); hdf = pd.DataFrame(S.hist); hdf.to_csv(os.path.join(R["metrics"], "training_history.csv"), index=False)
        hdf[["epoch"] + [c for c in hdf.columns if c.startswith("val_") or c in ("weights_used", "raw_val_auc", "ema_val_auc")]].to_csv(os.path.join(R["metrics"], "validation_metrics.csv"), index=False)
        save_full_ckpt(os.path.join(R["ckpt"], "last.pth"), S, tag="epoch_end"); save_full_ckpt(os.path.join(R["ckpt"], f"ckpt_epoch{epoch+1:03d}.pth"), S, tag="epoch_end")
        old = os.path.join(R["ckpt"], f"ckpt_epoch{epoch+1-cfg['KEEP_EPOCH_CKPTS']:03d}.pth")
        if os.path.exists(old): os.remove(old)
        if improved:
            atomic_save(dict(model=student.state_dict(), ema_model=ema.m.state_dict() if ema else None, best_source=src, epoch=epoch + 1, val_auc=v["auc"], config=cfg_to_jsonable(cfg), model_tag=tag,
                             history=S.hist, split_signature=sig, val_thr_opt=v["thr_opt"], val_aggregation=cfg["VIDEO_AGGREGATION"]), os.path.join(R["ckpt"], "best_auc.pth"))
        rem = cfg["MAX_SESSION_MINUTES"] - elapsed_min()
        print(f"\nEpoch {epoch+1}/{cfg['NUM_EPOCHS']}\n  Train Loss: {tr['loss']:.4f} (ce {tr['ce']:.3f} kd {tr['kd']:.3f} tc {tr['tc']:.4f} rec {tr['rec']:.4f})  Train Acc: {tr_acc:.4f}\n"
              f"  Effective generative weights (end of epoch): L_REC x {w_rec_end:.4f}  L_KL x {w_kl_end:.6f}  (warm-up factor {gs1:.3f})\n"
              f"  Generative input alpha: {ga0:.3f} -> {ga_end:.3f}\n"
              f"  Val Loss: {v['loss']:.4f}  Val Acc: {v['acc']:.4f}  Val BalAcc: {v['bal_acc']:.4f}  Val F1: {v['f1']:.4f}  Val AUC: {v['auc']:.4f} ({src})  Val PR-AUC: {v['pr_auc']:.4f}\n"
              f"  Val @Youden thr {v['thr_opt']:.3f}: F1 {v['f1_opt']:.4f}  BalAcc {v['bal_acc_opt']:.4f} | AUC by aggregation: {fmt_agg(v)}\n"
              f"  Best AUC: {S.best['auc']:.4f} (epoch {S.best['epoch']}, {S.best['source']})  Best F1: {S.best['f1']:.4f}  Best BalAcc: {S.best['bal_acc']:.4f}{'  <-- NEW BEST' if improved else ''}\n"
              f"  Epoch Time: {row['epoch_time_min']:.1f} min (val {val_time:.1f})  Elapsed: {elapsed_min():.1f} min  Estimated remaining session budget: {rem:.1f} min\n", flush=True)
        log(f"Epoch {epoch+1}: per-manipulation val AUC (all real vs this fake): " + ", ".join(f"{g_}={a_:.3f}" for g_, a_ in v.get("gen_auc", {}).items()))
        if any(not math.isfinite(row[k]) for k in ("train_loss", "val_loss")): raise RuntimeError("Non-finite epoch loss.")
        if epoch + 1 - S.best["epoch"] >= cfg["EARLY_STOP_PATIENCE"]: S.finished = True; stop_reason = f"early stopping (no val-AUC gain for {cfg['EARLY_STOP_PATIENCE']} epochs)"; break
    if stop_reason is None: stop_reason = "all epochs completed"; S.finished = S.epochs_done >= cfg["NUM_EPOCHS"]
    # session end: checkpoint FIRST, then compact diagnostic if time allows
    log(f"Stopping training: {stop_reason}")
    sess = os.path.join(R["ckpt"], f"session_end_epoch{S.epochs_done:03d}.pth"); lastp = os.path.join(R["ckpt"], "last.pth")
    save_full_ckpt(sess, S, partial=S.partial, tag="SESSION_END" + ("_PARTIAL" if S.partial else ""))
    shutil.copyfile(sess, lastp + ".tmp"); os.replace(lastp + ".tmp", lastp)
    remaining = cfg["MAX_SESSION_MINUTES"] - elapsed_min(); est_val = est_val_minutes(S, cfg); vd = None
    if S.partial is None and S.hist and not S.hist[-1].get("val_skipped", False):
        r = S.hist[-1]; vd = dict(auc=r["val_auc"], f1=r["val_f1"], acc=r["val_acc"], bal_acc=r["val_bal_acc"], pr_auc=r["val_pr_auc"], loss=r["val_loss"]); log("Diagnostic: reusing the epoch-end validation (weights unchanged).")
    elif remaining >= est_val + cfg["DIAG_FINAL_MARGIN_MINUTES"]:
        log("Session-end compact diagnostic validation on FF++ val ..."); vd = validate(dp_student, val_ds, cfg)
    else: log(f"Diagnostic validation SKIPPED: only {remaining:.1f} min left (est. validation {est_val:.1f} min).")
    last = S.hist[-1] if S.hist else {}; nanv = float("nan"); g = lambda k: (vd or {}).get(k, nanv)
    drow = dict(epoch=S.epochs_done + (S.partial["step_in_epoch"] / steps_per_epoch if S.partial else 0.0), val_auc=g("auc"), val_f1=g("f1"), val_acc=g("acc"), val_bal_acc=g("bal_acc"), val_pr_auc=g("pr_auc"),
                elapsed_train_min=elapsed_min(), lr=opt.param_groups[-1]["lr"], best_auc=S.best["auc"], train_loss=last.get("train_loss", nanv), val_loss=g("loss"), stop_reason=stop_reason)
    dcsv = os.path.join(R["diag"], "session_diagnostic.csv"); pd.DataFrame([drow]).to_csv(dcsv, mode="a", header=not os.path.exists(dcsv), index=False)
    plot_training(S.hist, R["figs"])
    aucs = finite_vals(S.hist, "val_auc")[-4:]; dauc = (aucs[-1] - aucs[0]) if len(aucs) >= 2 else float("nan")
    part_txt = f" (+ partial: step {S.partial['step_in_epoch']}/{steps_per_epoch})" if S.partial else ""
    print("\n" + "=" * 78 + "\nTRAINING SESSION COMPLETE" + f"\n  Stop reason: {stop_reason}\n  Current epoch: {S.epochs_done}{part_txt} of {cfg['NUM_EPOCHS']}\n"
          f"  Best validation AUC: {S.best['auc']:.4f} (epoch {S.best['epoch']}, weights={S.best['source']})\n  Last validation AUC: {g('auc'):.4f}\n  AUC change over last {len(aucs)} validated epochs: {dauc:+.4f}\n"
          f"  Current LR: {opt.param_groups[-1]['lr']:.3e}\n  Elapsed: {elapsed_min():.1f} min\n  Session-end ckpt: {sess}\n  Last checkpoint:  {lastp}\n  Best checkpoint:  {os.path.join(R['ckpt'], 'best_auc.pth')}\n"
          f"  Resume from:      {sess if not S.finished else '(training finished — evaluate best_auc.pth)'}\n  Session diagnostics: {dcsv}\n  Diagnostic (not a research conclusion): {trend_message(S.hist)}\n" + "=" * 78, flush=True)
    if S.finished: print("NEXT: set MODE='evaluate' and BEST_CHECKPOINT to best_auc.pth for the full paper-level evaluation.")
    else: print("NEXT SESSION: add this run's output as a Kaggle input dataset, set RESUME_CHECKPOINT to the session-end path above, keep all other CONFIG values unchanged.")

# =====================================================================================
#                                      EVALUATION
# =====================================================================================
def model_complexity(student, cfg, dev, ckpt_path):
    rows = {"params_total": sum(p.numel() for p in student.parameters()), "params_trainable": sum(p.numel() for p in student.parameters() if p.requires_grad),
            "checkpoint_MB": ckpt_size_mb(ckpt_path), "GFLOPs_per_clip": estimate_gflops(student, cfg, dev)}
    x = torch.rand(1, cfg["NUM_FRAMES"], 3, cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"], device=dev); student.eval()
    with torch.no_grad():
        for _ in range(3): student(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10): student(x)
        torch.cuda.synchronize(); dt = (time.time() - t) / 10
    rows.update(sec_per_clip=dt, ms_per_frame=1000 * dt / cfg["NUM_FRAMES"], fps=cfg["NUM_FRAMES"] / dt, clip_frames=cfg["NUM_FRAMES"]); return rows

def export_visual_analysis(student, videos, cfg, out_dir, dev, n_each=3):
    sel = [v for v in videos if v["label"] == 0][:n_each] + [v for v in videos if v["label"] == 1][:n_each]; ds = ClipDataset(sel, cfg, False, 1); student.eval(); os.makedirs(out_dir, exist_ok=True)
    for i in range(len(ds)):
        x, y, _ = ds[i]; x01 = to_x01(x[None], dev)
        with torch.no_grad(): o = student(x01, extras=True)
        T = cfg["NUM_FRAMES"]; fig, axs = plt.subplots(3, T, figsize=(1.4 * T, 4.6)); fm = o["ex_freq"][0].mean(1).cpu().numpy() if "ex_freq" in o else None
        for t in range(T):
            axs[0, t].imshow(x[t].numpy()); axs[0, t].axis("off")
            if fm is not None: axs[1, t].imshow(fm[t], cmap="magma")
            axs[1, t].axis("off")
        gate = o["gate"][0].cpu().numpy(); ax = fig.add_subplot(3, 1, 3); [a_.remove() for a_ in axs[2]]
        for nm, k in GATE_IDX.items():
            if nm in student.streams: ax.plot(range(T), gate[:, k], "o-", label=f"{nm} gate")
        if "ex_attn" in o: ax.bar(range(T), o["ex_attn"][0, 0, 1:].cpu().numpy(), alpha=0.35, label="CLS->frame attention (layer 1)")
        ax.set_xlabel("Frame"); ax.legend(fontsize=7); ax.set_title(f"{'Real' if int(y)==0 else 'Fake'} | {sel[i]['generator']}", fontsize=9)
        savefig(fig, os.path.join(out_dir, f"visual_{i:02d}"))

def run_evaluate(cfg, R, dev):
    path = cfg["BEST_CHECKPOINT"]
    if not path or not os.path.exists(path): raise FileNotFoundError(f"BEST_CHECKPOINT not found: '{path}'")
    ck = load_ckpt(path); ecfg = dict(cfg); cc = ck["config"]; NORM["type"] = cc.get("NORM_TYPE", "bn")
    for k in arch_keys() + ["TEMPORAL_STRIDE", "TEMPORAL_SAMPLING", "USE_KD", "DROPOUT", "USE_AMP", "GEN_FEED", "GEN_RES_DETACH"]:     # ARCHITECTURE COMES FROM THE CHECKPOINT
        if k in cc: v = cc[k]; ecfg[k] = tuple(v) if isinstance(v, list) else v
    ecfg["PRETRAINED"] = False      # weights come from the checkpoint (no download needed, not a fallback)
    student = StudentModel(ecfg).to(dev)
    raw = ck["model"]; ema_sd = ck.get("ema_model") or ((ck.get("ema") or {}).get("model") if isinstance(ck.get("ema"), dict) else None)
    want = cfg["EVAL_WEIGHTS"]; src = ck.get("best_source", "raw") if want == "best" else want
    if src == "ema" and ema_sd is None: raise RuntimeError("EVAL_WEIGHTS selects EMA but the checkpoint has no EMA weights.")
    student.load_state_dict(ema_sd if src == "ema" else raw, strict=True); student.eval(); tag = model_tag(ecfg); ckname = os.path.basename(path)
    log(f"Evaluating {path} | weights = {src.upper()} | epoch {ck.get('epoch')} | model {tag} (architecture from checkpoint config) | aggregation={cfg['VIDEO_AGGREGATION']}")
    print_model_summary(ecfg, student, dev); write_ablation_csv(ecfg, R, tag)
    dp = make_parallel(student, cfg); setup_mpl()
    ffpp = build_videos("FF++", cfg["FFPP_PATH"], "ffpp", cfg["FFPP_GROUPING_MODE"], seed=cfg["SEED"], cfg=cfg); splits, sig = ffpp_split(ffpp, cfg, R)
    if ck.get("split_signature") and ck["split_signature"] != sig: raise RuntimeError("FF++ split signature differs from the training run — refusing to evaluate on a different split.")
    def cap(v, name):
        if cfg["MAX_EVAL_VIDEOS"] and len(v) > cfg["MAX_EVAL_VIDEOS"]: v = random.Random(cfg["SEED"]).sample(v, cfg["MAX_EVAL_VIDEOS"]); log(f"{name}: capped to {len(v)} videos")
        return v
    mode_of = {"ffpp": cfg["FFPP_GROUPING_MODE"], "celebdf": cfg["CELEBDF_GROUPING_MODE"], "dfdc": cfg["DFDC_GROUPING_MODE"], "diffusion": "single_image"}
    sets = {"ffpp": ("FF++", splits["test"], "test"),
            "celebdf": ("Celeb-DF", cap(build_videos("CDF", cfg["CELEBDF_PATH"], "ffpp", cfg["CELEBDF_GROUPING_MODE"], seed=cfg["SEED"], cfg=cfg), "CDF"), "cross-dataset"),
            "dfdc": ("DFDC", cap(build_videos("DFDC", cfg["DFDC_PATH"], "dfdc", cfg["DFDC_GROUPING_MODE"], seed=cfg["SEED"]), "DFDC"), "cross-dataset")}
    if cfg["DIFFUSION_PATH"] and os.path.exists(cfg["DIFFUSION_PATH"]):
        sets["diffusion"] = ("Diffusion", build_videos("DIFF", cfg["DIFFUSION_PATH"], "ffpp", "single_image", cfg["DIFFUSION_MAX_PER_FOLDER"], cfg["SEED"], cfg=cfg), "cross-dataset (image-level)")
    else: log("Diffusion dataset is OPTIONAL and not available/not set -> skipped (main FF++/Celeb-DF/DFDC evaluation unaffected).")
    grouping_check(splits["val"], "FF++ val", cfg, cfg["FFPP_GROUPING_MODE"])
    for k, (n, v, _) in sets.items(): describe(v, n); grouping_check(v, n, cfg, mode_of[k])
    log("Deriving thresholds (Youden's J) on FF++ VALIDATION only ...")
    vds = ClipDataset(splits["val"], ecfg, False, cfg["MAX_EVAL_CLIPS"]); fv, vv, _ = run_inference(dp, vds, ecfg, "val-thr")
    thr_v, thr_f = youden_threshold(vv["label"].values, vv["p_video"].values), youden_threshold(fv["label"].values, fv["p_frame"].values)
    vv.to_csv(os.path.join(R["pred"], "video_predictions_val.csv"), index=False)      # validation scores: threshold-stability / temperature analyses
    pd.DataFrame([dict(source="FF++ val", video_threshold=thr_v, frame_threshold=thr_f, video_aggregation=cfg["VIDEO_AGGREGATION"])]).to_csv(os.path.join(R["metrics"], "validation_thresholds.csv"), index=False)
    log(f"Thresholds (from FF++ val only): video={thr_v:.4f} frame={thr_f:.4f}")
    summary, boots, cms, gates, res = [], [], [], [], {}
    active = {"spatial": True, "artifact": ecfg["USE_SPATIAL_ARTIFACT"], "frequency": ecfg["USE_FREQUENCY"]}
    for key, (name, vids, split) in sets.items():
        ds = ClipDataset(vids, ecfg, False, cfg["MAX_EVAL_CLIPS"]); fdf, vdf, _ = run_inference(dp, ds, ecfg, name)
        for df, lvl, thr, col in ((fdf, "frame", thr_f, "p_frame"), (vdf, "video", thr_v, "p_video")):
            df["predicted_label"] = (df[col] >= thr).astype(int); df.insert(0, "dataset", name); df["split"] = split; df["checkpoint"] = ckname; df["model_config"] = f"{tag}|{src}"
            df.rename(columns={"label": "true_label", col: "predicted_probability"}, inplace=True)
            df.to_csv(os.path.join(R["pred"], f"{lvl}_predictions_{key}.csv"), index=False)
            m = compute_metrics(df["true_label"].values, df["predicted_probability"].values, thr); summary.append(dict(dataset=name, level=lvl, **m)); cms.append(dict(dataset=name, level=lvl, threshold=thr, tn=m["tn"], fp=m["fp"], fn=m["fn"], tp=m["tp"]))
            if lvl == "video":
                ci = bootstrap_ci(df["true_label"].values, df["predicted_probability"].values, thr, cfg["BOOTSTRAP_ITERS"], cfg["SEED"])
                for mk, (lo, hi, nv) in ci.items(): boots.append(dict(dataset=name, level="video", metric=mk, estimate=m[mk], ci_low=lo, ci_high=hi, n_videos=len(df), valid_iters=nv, threshold=thr))
            if lvl == "video":
                for grp, sub in (("all", df), ("real", df[df.true_label == 0]), ("fake", df[df.true_label == 1])):
                    for br in ("spatial", "artifact", "frequency"):
                        if active[br] and len(sub):
                            c_ = sub[f"gate_{br}"]; gates.append(dict(dataset=name, group=grp, branch=br, n=len(sub), gate_mean=c_.mean(), gate_std=c_.std(), gate_p10=c_.quantile(.1), gate_p50=c_.quantile(.5), gate_p90=c_.quantile(.9)))
        if key in ("ffpp", "diffusion"):
            real = vdf[vdf.true_label == 0]
            for gname in sorted(set(vdf[vdf.true_label == 1]["generator"])):
                sub = pd.concat([real, vdf[(vdf.true_label == 1) & (vdf.generator == gname)]]); mm = compute_metrics(sub["true_label"].values, sub["predicted_probability"].values, thr_v)
                summary.append(dict(dataset=f"{name}[{gname}]", level="video (real vs this)", **mm))
        res[key] = (name, fdf, vdf, thr_f, thr_v); log(f"{name}: video AUC={[s for s in summary if s['dataset']==name and s['level']=='video'][0]['auc']:.4f}")
    S_df = pd.DataFrame(summary); S_df.to_csv(os.path.join(R["metrics"], "final_metrics_summary.csv"), index=False)
    B_df = pd.DataFrame(boots); B_df.to_csv(os.path.join(R["metrics"], "bootstrap_confidence_intervals.csv"), index=False)
    pd.DataFrame(cms).to_csv(os.path.join(R["metrics"], "confusion_matrix_values.csv"), index=False); pd.DataFrame(gates).to_csv(os.path.join(R["metrics"], "fusion_gate_statistics.csv"), index=False)
    cx = model_complexity(student, ecfg, dev, path); pd.DataFrame([cx]).to_csv(os.path.join(R["metrics"], "model_complexity.csv"), index=False)
    plot_training(ck.get("history", []), R["figs"])
    for key, (name, fdf, vdf, tf, tv) in res.items():
        ser = {"Video": (vdf["true_label"].values, vdf["predicted_probability"].values), "Frame": (fdf["true_label"].values, fdf["predicted_probability"].values)}
        plot_curve("roc", ser, f"ROC — {name}", os.path.join(R["figs"], f"roc_{key}")); plot_curve("pr", ser, f"Precision–Recall — {name}", os.path.join(R["figs"], f"pr_{key}"))
        plot_cm([s for s in summary if s["dataset"] == name and s["level"] == "video"][0], f"{name} (video-level)", os.path.join(R["figs"], f"confusion_{key}"))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for key, (name, _, vdf, _, _) in res.items():
        y, p = vdf["true_label"].values, vdf["predicted_probability"].values
        if len(np.unique(y)) > 1: fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("Video-level ROC across datasets"); ax.legend(loc="lower right", fontsize=8); savefig(fig, os.path.join(R["figs"], "roc_all_datasets"))
    # reliability diagram: same bins, same threshold and same ECE as final_metrics_summary.csv (video level)
    ece_of = {name: [s for s in summary if s["dataset"] == name and s["level"] == "video"][0]["ece"] for _, (name, _, _, _, _) in res.items()}
    plot_reliability([(name, vdf["true_label"].values, vdf["predicted_probability"].values, tv) for _, (name, _, vdf, _, tv) in res.items()], ece_of, os.path.join(R["figs"], "calibration_reliability"))
    if cfg["SAVE_VISUAL_ANALYSIS"]: export_visual_analysis(student, splits["test"], ecfg, os.path.join(R["figs"], "visual_analysis"), dev)
    main_ = S_df[~S_df["level"].str.contains("real vs")].copy(); ci_txt = {(r.dataset, r.metric): f"[{r.ci_low:.3f},{r.ci_high:.3f}]" for _, r in B_df.iterrows()}
    cols = ["dataset", "level", "acc", "bal_acc", "precision", "recall", "specificity", "f1", "mcc", "auc", "pr_auc", "eer", "ece"]
    ren = {"dataset": "Dataset", "level": "Level", "acc": "Acc", "bal_acc": "BalAcc", "precision": "Prec", "recall": "Recall", "specificity": "Spec", "f1": "F1", "mcc": "MCC", "auc": "ROC-AUC", "pr_auc": "PR-AUC", "eer": "EER", "ece": "ECE"}
    fmt = lambda v: f"{v:.4f}"
    print("\n" + "=" * 130 + f"\nFINAL RESULTS | model {tag} | weights {src.upper()} | ckpt {ckname} | aggregation {cfg['VIDEO_AGGREGATION']} | thresholds (FF++ val only): video {thr_v:.4f}, frame {thr_f:.4f}\n" + "=" * 130)
    print("PRIMARY — video-level:"); print(main_[main_.level == "video"][cols].rename(columns=ren).to_string(index=False, float_format=fmt))
    print("\nSUPPORTING — frame-level:"); print(main_[main_.level == "frame"][cols].rename(columns=ren).to_string(index=False, float_format=fmt))
    print("\nVideo-level 95%% bootstrap CIs (resampling videos, %d iters):" % cfg["BOOTSTRAP_ITERS"])
    for ds_name in (B_df["dataset"].unique() if len(B_df) else []):
        print(f"  {ds_name:10s} " + "  ".join(f"{m}={B_df[(B_df.dataset==ds_name)&(B_df.metric==m)].estimate.values[0]:.4f} {ci_txt[(ds_name, m)]}" for m in ("auc", "f1", "acc", "bal_acc")))
    print(f"\nComplexity: params {cx['params_total']/1e6:.2f}M | ckpt {cx['checkpoint_MB']:.0f} MB | {cx['GFLOPs_per_clip']:.1f} GFLOPs/clip | {cx['ms_per_frame']:.1f} ms/frame | {cx['fps']:.1f} FPS")
    print(f"All outputs in: {R['run']}\nEVALUATION COMPLETE")

# =====================================================================================
#                                        MAIN
# =====================================================================================
class TeacherAsModel(nn.Module):
    """Adapter so run_inference() can evaluate the frozen teacher with the SAME clips / aggregation as the student."""
    def __init__(self, teacher, amp):
        super().__init__(); self.t, self.amp = teacher, amp
    def forward(self, x01, **kw):
        B, T = x01.shape[:2]
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp): tl = self.t(x01.flatten(0, 1)).float().view(B, T, 2)
        pv = F.softmax(tl, -1)[..., 1].mean(1)                                     # video prob = mean of frame probs (same as eval_teacher)
        vl = torch.log(torch.stack([1 - pv, pv], -1).clamp_min(1e-7))              # softmax(vl) == [1-pv, pv]
        return dict(video_logits=vl, frame_logits=tl, gate=torch.zeros(B, T, 3, device=x01.device))

def run_eval_teacher(cfg, R, dev):
    """Teacher-only cross-dataset evaluation (video level). Threshold from FF++ val only, as for the student."""
    teacher = load_teacher(cfg, dev); model = TeacherAsModel(teacher, cfg["USE_AMP"]).eval()
    ffpp = build_videos("FF++", cfg["FFPP_PATH"], "ffpp", cfg["FFPP_GROUPING_MODE"], seed=cfg["SEED"], cfg=cfg); splits, _ = ffpp_split(ffpp, cfg, R)
    sets = {"FF++": splits["test"],
            "Celeb-DF": build_videos("CDF", cfg["CELEBDF_PATH"], "ffpp", cfg["CELEBDF_GROUPING_MODE"], seed=cfg["SEED"], cfg=cfg),
            "DFDC": build_videos("DFDC", cfg["DFDC_PATH"], "dfdc", cfg["DFDC_GROUPING_MODE"], seed=cfg["SEED"])}
    if cfg["DIFFUSION_PATH"] and os.path.exists(cfg["DIFFUSION_PATH"]):
        sets["Diffusion"] = build_videos("DIFF", cfg["DIFFUSION_PATH"], "ffpp", "single_image", cfg["DIFFUSION_MAX_PER_FOLDER"], cfg["SEED"], cfg=cfg)
    log("Teacher: deriving threshold (Youden's J) on FF++ VALIDATION only ...")
    _, vv, _ = run_inference(model, ClipDataset(splits["val"], cfg, False, cfg["MAX_EVAL_CLIPS"]), cfg, "teacher val-thr", collect_frames=False)
    thr = youden_threshold(vv["label"].values, vv["p_video"].values); log(f"Teacher video threshold (FF++ val) = {thr:.4f}")
    student_kd0 = {"FF++": 0.9829, "Celeb-DF": 0.7564, "DFDC": 0.6554, "Diffusion": 0.6410}      # your abl_T1F1A1KD0 video AUCs, for side-by-side comparison
    rows = []
    for name, vids in sets.items():
        _, vdf, _ = run_inference(model, ClipDataset(vids, cfg, False, cfg["MAX_EVAL_CLIPS"]), cfg, f"teacher {name}", collect_frames=False)
        y, p = vdf["label"].values, vdf["p_video"].values; m = compute_metrics(y, p, thr); lo, hi, _ = bootstrap_ci(y, p, thr, cfg["BOOTSTRAP_ITERS"], cfg["SEED"])["auc"]
        rows.append(dict(dataset=name, teacher_auc=m["auc"], auc_ci_low=lo, auc_ci_high=hi, student_KD0_auc=student_kd0.get(name), eer=m["eer"], bal_acc=m["bal_acc"], recall=m["recall"], specificity=m["specificity"], ece=m["ece"], n_videos=len(vdf)))
        vdf.to_csv(os.path.join(R["pred"], f"teacher_video_predictions_{name}.csv"), index=False); log(f"Teacher {name}: video AUC={m['auc']:.4f} [{lo:.3f},{hi:.3f}]")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R["metrics"], "teacher_cross_dataset.csv"), index=False)
    print("\n" + "=" * 100 + "\nTEACHER (DP-ViT) vs STUDENT KD0 — video-level, threshold from FF++ val\n" + "=" * 100)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}")); print("TEACHER EVALUATION COMPLETE")

def main():
    cfg = dict(CONFIG)
    ov = os.environ.get("GC_OVERRIDES")                      # JSON dict set by the launcher: lets several processes run different configs from ONE script file
    if ov:
        ov = json.loads(ov)
        for k, v in ov.items():
            if k not in cfg: raise KeyError(f"GC_OVERRIDES key '{k}' is not in CONFIG")
            cfg[k] = tuple(v) if isinstance(v, list) else v
        print("CONFIG overrides:", ov, flush=True)
    if os.environ.get("GC_EXTRACT_ONLY") == "1":              # launcher extracts the faces ONCE before starting parallel runs (avoids two processes unzipping at the same time)
        check_faces(); print("faces check done", flush=True); return
    assert cfg["MODE"] in ("train", "evaluate", "eval_teacher"); NORM["type"] = cfg.get("NORM_TYPE", "gn")
    assert torch.cuda.is_available(), "No GPU — enable a Kaggle GPU accelerator."
    if cfg["NUM_FRAMES"] > 64: raise ValueError("NUM_FRAMES <= 64")
    if cfg["TEMPORAL_SAMPLING"] not in ("window", "contiguous"): raise ValueError("TEMPORAL_SAMPLING must be 'window' or 'contiguous'")
    if cfg["TRAIN_WINDOW"] < cfg["NUM_FRAMES"] or cfg["EVAL_WINDOW"] < cfg["NUM_FRAMES"]: raise ValueError("TRAIN_WINDOW and EVAL_WINDOW must be >= NUM_FRAMES")
    for k in ("FFPP_GROUPING_MODE", "CELEBDF_GROUPING_MODE", "DFDC_GROUPING_MODE"):
        if cfg[k] not in GROUP_MODES: raise ValueError(f"{k} must be one of {GROUP_MODES}, got '{cfg[k]}'")
    if cfg["FFPP_GROUPING_MODE"] == "single_image": raise ValueError("FFPP_GROUPING_MODE='single_image' is invalid: FF++ needs video-level grouping for the official split.")
    if cfg["GENERATIVE_WARMUP_EPOCHS"] < 0: raise ValueError("GENERATIVE_WARMUP_EPOCHS must be >= 0")
    if cfg["GENERATIVE_INPUT_WARMUP_EPOCHS"] < 0: raise ValueError("GENERATIVE_INPUT_WARMUP_EPOCHS must be >= 0")
    if cfg["BATCH_SIZE"] != "auto": cfg["BATCH_SIZE"] = int(cfg["BATCH_SIZE"])      # accepts 2 or "2"
    exp = f"genconvit_{model_tag(cfg)}" if cfg["EXP_NAME"] == "auto" else cfg["EXP_NAME"]
    run = os.path.join(cfg["OUTPUT_DIR"], exp); R = dict(run=run, exp=exp)
    for k, d in dict(ckpt="checkpoints", logs="logs", metrics="metrics", pred="predictions", figs="figures", configs="configs", diag="diagnostics").items(): R[k] = os.path.join(run, d); os.makedirs(R[k], exist_ok=True)
    LOGFILE["path"] = os.path.join(R["logs"], f"{cfg['MODE']}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    set_seed(cfg["SEED"]); torch.backends.cudnn.benchmark = True; dev = torch.device("cuda")
    R["env"] = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, timm=timm.__version__, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], n_gpus=torch.cuda.device_count())
    log(f"Experiment: {exp} | MODE={cfg['MODE']} | {R['env']}")
    log(f"Ablation: ARTIFACT={cfg['USE_SPATIAL_ARTIFACT']} FREQUENCY={cfg['USE_FREQUENCY']} TEMPORAL={cfg['USE_TEMPORAL']} KD={cfg['USE_KD']} | variant={cfg['GENCONVIT_VARIANT']} | T={cfg['NUM_FRAMES']}")
    log(f"Backbone note: {GENCONVIT_DETAIL}")
    check_faces()
    need = ["FFPP_PATH"] + (["CELEBDF_PATH", "DFDC_PATH"] if cfg["MODE"] in ("evaluate", "eval_teacher") else [])
    for k in need:
        if not os.path.exists(cfg[k]): raise FileNotFoundError(f"{k} does not exist: {cfg[k]}")
    if cfg["MODE"] == "train": run_train(cfg, R, dev)
    elif cfg["MODE"] == "eval_teacher": run_eval_teacher(cfg, R, dev)
    else: run_evaluate(cfg, R, dev)

main()


2nd Cell:

%%bash
M=/kaggle/input/models/syedazmulhasansabbir
CK43=$M/genconvit-nosbi-s43/tensorflow2/default/1/best_auc.pth
CK44=$M/genconvit-nosbi-s44/tensorflow2/default/1/best_auc.pth

# ---- 0) pre-flight: show what actually exists ----
for p in "$CK43" "$CK44"; do [ -e "$p" ] && echo "OK       $p ($([ -d "$p" ] && echo folder || echo file))" || echo "MISSING  $p"; done
echo "--- all best_auc.pth visible under /kaggle/input:"
find /kaggle/input -name "best_auc.pth" 2>/dev/null
echo "--- official_splits caches visible (if any):"
find /kaggle/input /kaggle/working -type d -name official_splits 2>/dev/null

# If the line above lists a folder, put its PARENT here (the folder that contains train.json/val.json/test.json), else leave "".
SPLIT_DIR=""
SPLIT_JSON=""; [ -n "$SPLIT_DIR" ] && SPLIT_JSON=",\"FFPP_SPLIT_DIR\":\"$SPLIT_DIR\""

run() {  # $1 = GPU id, $2 = run name, $3 = checkpoint path
  CUDA_VISIBLE_DEVICES=$1 GC_OVERRIDES="{\"MODE\":\"evaluate\",\"EXP_NAME\":\"$2\",\"NUM_WORKERS_EVAL\":2,\"BEST_CHECKPOINT\":\"$3\"$SPLIT_JSON}" \
    python /kaggle/working/genconvit_tfkd.py > /kaggle/working/log_$2.txt 2>&1
}

run 0 eval_noSBI_s43 "$CK43" & P0=$!
run 1 eval_noSBI_s44 "$CK44" & P1=$!
wait $P0; S0=$?
wait $P1; S1=$?
echo "exit codes: s43=$S0  s44=$S1"

# show the real error if a run failed, otherwise the result table
for pair in "eval_noSBI_s43:$S0" "eval_noSBI_s44:$S1"; do
  n=${pair%%:*}; st=${pair##*:}
  echo "=========== $n (exit $st)"
  if [ "$st" -ne 0 ]; then
    tail -n 30 /kaggle/working/log_$n.txt
  else
    grep -A 12 "PRIMARY" /kaggle/working/log_$n.txt | head -n 14
  fi
done
# copy results into one flat folder, adding the suffix _noSBI_S43 / _noSBI_S44 to every file name
OUT=/kaggle/working/results_noSBI
rm -rf $OUT; mkdir -p $OUT
for pair in "eval_noSBI_s43:noSBI_S43" "eval_noSBI_s44:noSBI_S44"; do
  run_dir=/kaggle/working/output/${pair%%:*}
  suf=${pair##*:}
  for sub in metrics predictions figures; do
    for f in "$run_dir/$sub"/*; do
      [ -f "$f" ] || continue
      b=$(basename "$f"); ext="${b##*.}"; stem="${b%.*}"
      mkdir -p "$OUT/$sub"
      cp "$f" "$OUT/$sub/${stem}_${suf}.${ext}"
    done
  done
  cp /kaggle/working/log_${pair%%:*}.txt "$OUT/log_${suf}.txt"
done