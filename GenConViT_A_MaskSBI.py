%%bash
# =====================================================================================================================
# CELL 2 -- ARM A = boundary-supervised SBI (C1): SBI_PROB 0.3 + blend-mask head      *** ONE SEED PER KAGGLE NOTEBOOK SESSION ***
# Session 1: SEED=42   |   Session 2: SEED=43   (the ONLY line that differs is the first assignment below).
# Each session is fully independent: [setup + self-test] -> TRAIN (this seed, GPU 0, 4 workers) -> EVALUATE (FF++ / Celeb-DF / DFDC / Diffusion + C3 conformal) -> ZIP.
# Time plan (720 min Kaggle limit):  setup <=15 | training soft stop ~532 (MAX_SESSION_MINUTES 560 - reserve) / HARD kill at 585 | evaluation HARD kill 80 | zip ~5  => worst case ~685 min.
# Needs Cell 1 (%%writefile genconvit_tfkd.py) to have been run in THIS session first.
# =====================================================================================================================
SEED=42                      # <<< ONLY LINE TO CHANGE: 42 in notebook session 1, 43 in notebook session 2
NUM_EPOCHS=14; CLIPS=3200    # keep IDENTICAL in both sessions (3200 clips/epoch, 14 epochs = same recipe as the earlier SBI runs)
SOFT_MIN=560                 # soft stop of training (script stops after the current step, saves a checkpoint, runs a compact diagnostic)
TRAIN_HARD_MIN=585           # HARD kill of training (a hung process can no longer run past this)
EVAL_HARD_MIN=80             # HARD kill of evaluation; evaluation is skipped if it could not finish before SESSION_LIMIT_MIN
SESSION_LIMIT_MIN=700

[ -f /kaggle/working/genconvit_tfkd.py ] || { echo "MISSING genconvit_tfkd.py -> run the %%writefile script cell (Cell 1) first"; exit 1; }
T0=$(date +%s); W=/kaggle/working; OUT=$W/output; SCRIPT=$W/genconvit_nov.py; PFX=A_MaskSBI; RUN=s${SEED}_${PFX}; CK=$OUT/$RUN/checkpoints; mkdir -p $W/nov
echo "RUN=$RUN | SEED=$SEED | epochs=$NUM_EPOCHS | clips/epoch=$CLIPS"

cat > $W/nov/nov_patch.dsl <<'NOV_DSL_EOF'
@@PATCH config_keys
    "FREQ_TYPE": "fft",
}
@@WITH
    "FREQ_TYPE": "fft",
    # ================= NOVELTY PATCH (C1 mask head | C2 FD-SBI | C3 conformal) =================
    "FDSBI_PROB": 0.0,                    # fraction of REAL training clips turned into FREQUENCY-domain pseudo-fakes (label 1); SBI_PROB + FDSBI_PROB <= 1
    "USE_MASK_HEAD": False,               # C1: auxiliary head on the artifact-stream map predicts the blurred SBI blend mask
    "LAMBDA_MASK": 0.5,                   # weight of the mask loss (BCE + Dice)
    "CONFORMAL_ALPHAS": (0.01, 0.02, 0.05, 0.10),   # C3: target false-positive rates on REAL videos for the conformal thresholds
}
@@END
@@PATCH sbi_mask_split
        S = self.S; yy, xx = np.mgrid[0:S, 0:S].astype(np.float32); x = xx / S - q["cx"]; y = yy / S - q["cy"]
        c, s = math.cos(q["rot"]), math.sin(q["rot"]); xr = c * x + s * y; yr = -s * x + c * y; th = np.arctan2(yr, xr)
        rad = 1.0 + q["a1"] * np.sin(th + q["p1"]) + q["a2"] * np.sin(2 * th + q["p2"])
        m = (((xr / q["rx"]) ** 2 + (yr / q["ry"]) ** 2) <= rad ** 2).astype(np.uint8) * 255
        mk = np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(max(1.0, q["msig"] * S))), dtype=np.float32)[..., None] / 255.0 * q["alpha"]
@@WITH
        S = self.S; mk = self._sbi_mask(q)[..., None] * q["alpha"]
@@END
@@PATCH dataset_methods
    def _prep(self, im, p, rng, k=0):
@@WITH
    def _sbi_mask(self, q):
        """Blurred elliptical blend mask in [0,1] WITHOUT alpha (same geometry as the SBI blend; also the C1 boundary-head target)."""
        S = self.S; yy, xx = np.mgrid[0:S, 0:S].astype(np.float32); x = xx / S - q["cx"]; y = yy / S - q["cy"]
        c, s = math.cos(q["rot"]), math.sin(q["rot"]); xr = c * x + s * y; yr = -s * x + c * y; th = np.arctan2(yr, xr)
        rad = 1.0 + q["a1"] * np.sin(th + q["p1"]) + q["a2"] * np.sin(2 * th + q["p2"])
        m = (((xr / q["rx"]) ** 2 + (yr / q["ry"]) ** 2) <= rad ** 2).astype(np.uint8) * 255
        return np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(max(1.0, q["msig"] * S))), dtype=np.float32) / 255.0
    def _make_fdsbi(self, rng):
        """ONE set of spectral-edit parameters per CLIP (static across frames, like SBI). Always draws the same number of random values."""
        return dict(mode=("atten", "boost", "peaks", "replica")[int(rng.integers(0, 4))], r0=float(rng.uniform(0.18, 0.55)), rw=float(rng.uniform(0.05, 0.15)), s=float(rng.uniform(0.5, 1.2)),
                    k=int(rng.integers(1, 4)), pr=[float(rng.uniform(0.25, 0.85)) for _ in range(3)], pa=[float(rng.uniform(0.0, math.pi)) for _ in range(3)], psig=float(rng.uniform(0.012, 0.03)),
                    shx=int(rng.integers(12, 37)), shy=int(rng.integers(-36, 37)), beta=float(rng.uniform(0.15, 0.45)))
    def _nov_aux(self, p, kind):
        """kind: 0 real, 1 FF++ fake, 2 SBI, 3 FD-SBI. Mask target at S/8 resolution (= the artifact-stream map); valid only for REAL (all-zero target) and SBI clips."""
        h = self.S // 8
        if kind == 2: m = self._sbi_mask(p[4]["sbi"]).reshape(h, 8, h, 8).mean((1, 3)); valid = 1.0
        else: m = np.zeros((h, h), np.float32); valid = 1.0 if kind == 0 else 0.0
        return dict(mask=torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32)), mvalid=torch.tensor(valid), kind=torch.tensor(int(kind)))
    def _prep(self, im, p, rng, k=0):
@@END
@@PATCH prep_fdsbi
            if aug.get("sbi") is not None: im = self._self_blend(im, aug["sbi"])
@@WITH
            if aug.get("sbi") is not None: im = self._self_blend(im, aug["sbi"])
            if aug.get("fdsbi") is not None: im = nov_fd_blend(im, aug["fdsbi"])
@@END
@@PATCH getitem_init
vi, w0, W = self.items[i]; v = self.v[vi]; rng = None; p = None; fk = False
@@WITH
vi, w0, W = self.items[i]; v = self.v[vi]; rng = None; p = None; fk = False; kind = 1 if v["label"] == 1 else 0
@@END
@@PATCH getitem_sbi_line
            if v["label"] == 0 and self.cfg.get("SBI_PROB", 0.0) > 0 and rng.random() < self.cfg["SBI_PROB"]: p[4]["sbi"] = self._make_sbi(rng); fk = True
@@WITH
            if v["label"] == 0 and (self.cfg.get("SBI_PROB", 0.0) > 0 or self.cfg.get("FDSBI_PROB", 0.0) > 0):
                u_ = rng.random(); sp_ = self.cfg.get("SBI_PROB", 0.0)                      # ONE uniform draw: [0,sp_) -> SBI, [sp_, sp_+fp_) -> FD-SBI (SBI-only runs keep their exact old draws)
                if u_ < sp_: p[4]["sbi"] = self._make_sbi(rng); fk = True; kind = 2
                elif u_ < sp_ + self.cfg.get("FDSBI_PROB", 0.0):
                    p[4]["fdsbi"] = self._make_fdsbi(rng); fk = True; kind = 3
                    p[4]["blur"] = None; p[4]["down"] = None                                # blur / down-scaling would erase the spectral edit while REAL clips keep them -> label conflict
@@END
@@PATCH getitem_return
        return torch.from_numpy(np.stack(frames)), torch.tensor(1 if fk else v["label"], dtype=torch.long), i
@@WITH
        x_ = torch.from_numpy(np.stack(frames)); y_ = torch.tensor(1 if fk else v["label"], dtype=torch.long)
        if not self.train: return x_, y_, i
        return x_, y_, i, self._nov_aux(p, kind)                                           # TRAIN items carry a 4th element: dict(mask, mvalid, kind)
@@END
@@PATCH artifact_encoder_forward
    def forward(self, x01): return self.pool(self.net(self.bn_in(self.residual(x01))))
@@WITH
    def forward(self, x01, want_map=False):
        fm = self.net(self.bn_in(self.residual(x01))); z = self.pool(fm)
        return (z, fm) if want_map else z
@@END
@@PATCH student_init
        if self.use_a: self.aenc = SpatialArtifactEncoder(D)
@@WITH
        if self.use_a: self.aenc = SpatialArtifactEncoder(D)
        self.mhead = nn.Sequential(nn.Conv2d(96, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 1, 1)) if (self.use_a and cfg.get("USE_MASK_HEAD", False)) else None     # C1 (96 = last width of SpatialArtifactEncoder)
@@END
@@PATCH student_forward_def
    def forward(self, x01, extras=False, debug=False, gen_alpha=1.0):
@@WITH
    def nov_aenc(self, x):
        if self.mhead is None: return self.aenc(x), None
        a, fm = self.aenc(x, want_map=True); return a, self.mhead(fm).float().squeeze(1)        # mask logits [N, S/8, S/8]
    def forward(self, x01, extras=False, debug=False, gen_alpha=1.0):
@@END
@@PATCH student_forward_init
            s, rec, kl = self.gen(x, gen_alpha); feats = [s]; a = f = None
@@WITH
            s, rec, kl = self.gen(x, gen_alpha); feats = [s]; a = f = None; mlog = None
@@END
@@PATCH student_forward_aenc
a = self.aenc(x); feats.append(a);
@@WITH
a, mlog = self.nov_aenc(x); feats.append(a);
@@END
@@PATCH student_forward_out
        out.update({"ex_" + k: v_ for k, v_ in ex.items()}); return out
@@WITH
        if mlog is not None: out["mask_logits"] = mlog.view(B, T, *mlog.shape[-2:])
        out.update({"ex_" + k: v_ for k, v_ in ex.items()}); return out
@@END
@@PATCH param_breakdown_row
("Spatial-artifact branch", ("aenc",)),
@@WITH
("Spatial-artifact branch", ("aenc",)), ("Boundary-mask head (C1)", ("mhead",)),
@@END
@@PATCH param_group
    if n.startswith(("aenc.", "fenc.", "gen.resenc.")): return "forensic"
@@WITH
    if n.startswith(("aenc.", "fenc.", "gen.resenc.", "mhead.")): return "forensic"
@@END
@@PATCH compute_loss_rename
def compute_loss(out, y, cfg, gscale=1.0, kscale=1.0):
@@WITH
def _compute_loss_orig(out, y, cfg, gscale=1.0, kscale=1.0):
@@END
@@PATCH pretrain_unpack
    for k, (x, _, _) in enumerate(tqdm(dl, total=steps, desc="gen-pretrain", leave=False)):
@@WITH
    for k, (x, _, _, _) in enumerate(tqdm(dl, total=steps, desc="gen-pretrain", leave=False)):
@@END
@@PATCH pretrain_clean_cfg
AUG_NOISE_PROB=0.0, SBI_PROB=0.0)   # reconstruct CLEAN frames
@@WITH
AUG_NOISE_PROB=0.0, SBI_PROB=0.0, FDSBI_PROB=0.0)   # reconstruct CLEAN frames
@@END
@@PATCH preflight_unpack
    x0, y0, _ = train_ds[0]; v0 = train_ds.v[train_ds.items[0][0]]
@@WITH
    x0, y0 = train_ds[0][:2]; v0 = train_ds.v[train_ds.items[0][0]]
@@END
@@PATCH train_loop_unpack
        for mb, (x, y, _) in enumerate(dl_it):
@@WITH
        for mb, (x, y, _, aux_) in enumerate(dl_it):
@@END
@@PATCH train_loop_dev
            x01 = to_x01(x, dev); y = y.to(dev, non_blocking=True)
            try:
@@WITH
            x01 = to_x01(x, dev); y = y.to(dev, non_blocking=True); aux_ = nov_aux_dev(aux_, dev)
            try:
@@END
@@PATCH train_loop_loss
L = compute_loss(out, y, cfg, gen_scale(cfg, S.global_step, steps_per_epoch), ks)
@@WITH
L = compute_loss(out, y, cfg, gen_scale(cfg, S.global_step, steps_per_epoch), ks, aux=aux_)
@@END
@@PATCH acc_keys
("loss", "ce", "kd", "tc", "rec", "correct", "n", "aux0", "aux1", "aux2")}
@@WITH
("loss", "ce", "kd", "tc", "rec", "correct", "n", "aux0", "aux1", "aux2", "mask")}
@@END
@@PATCH acc_update
                acc["rec"] += L["rec"].detach() * B_;
@@WITH
                acc["rec"] += L["rec"].detach() * B_; acc["mask"] += L["mask"].detach() * B_;
@@END
@@PATCH epoch_log
        log(f"Epoch {epoch+1}: TRAIN frame-accuracy per branch head: "
@@WITH
        nov_epoch_log(epoch, acc, cfg)
        log(f"Epoch {epoch+1}: TRAIN frame-accuracy per branch head: "
@@END
@@PATCH train_banner
    log(f"================ TRAINING START |
@@WITH
    nov_banner(cfg)
    log(f"================ TRAINING START |
@@END
@@PATCH eval_arch_keys
"GEN_FEED", "GEN_RES_DETACH"]:
@@WITH
"GEN_FEED", "GEN_RES_DETACH", "USE_MASK_HEAD"]:
@@END
@@PATCH eval_conformal_hook
    S_df = pd.DataFrame(summary); S_df.to_csv(
@@WITH
    nov_conformal_report(cfg, R, vv, res, thr_v)
    S_df = pd.DataFrame(summary); S_df.to_csv(
@@END
@@PATCH main_selftest
    if os.environ.get("GC_EXTRACT_ONLY") == "1":
@@WITH
    if os.environ.get("NOV_SELFTEST"): nov_selftest(int(os.environ["NOV_SELFTEST"])); return
    if os.environ.get("GC_EXTRACT_ONLY") == "1":
@@END
@@PATCH main_check
    assert cfg["MODE"] in ("train", "evaluate", "eval_teacher"); NORM["type"] = cfg.get("NORM_TYPE", "gn")
@@WITH
    assert cfg["MODE"] in ("train", "evaluate", "eval_teacher"); NORM["type"] = cfg.get("NORM_TYPE", "gn")
    nov_check_cfg(cfg)
@@END
@@PATCH helper_block
class TeacherAsModel(nn.Module):
@@WITH
# =====================================================================================
#   NOVELTY PATCH (C1 mask head | C2 frequency-domain SBI | C3 conformal thresholds)
#   C1  boundary-supervised SBI : auxiliary head on the artifact-stream map predicts the (blurred) parametric blend mask
#   C2  FD-SBI                  : self-blending in the Fourier-MAGNITUDE domain (phase of x kept), 4 spectral modes
#   C3  conformal thresholding  : split-conformal real-class threshold + class-conditional prediction sets (evaluation)
# =====================================================================================
def nov_check_cfg(cfg):
    if cfg.get("USE_MASK_HEAD") and not cfg["USE_SPATIAL_ARTIFACT"]: raise ValueError("USE_MASK_HEAD needs USE_SPATIAL_ARTIFACT=True (the mask head reads the artifact-stream feature map)")
    if cfg["IMAGE_SIZE"] % 8: raise ValueError("IMAGE_SIZE must be divisible by 8 (mask target is S/8 x S/8)")
    if cfg.get("SBI_PROB", 0.0) + cfg.get("FDSBI_PROB", 0.0) > 1.0: raise ValueError("SBI_PROB + FDSBI_PROB must be <= 1")
    if cfg.get("USE_MASK_HEAD") and cfg.get("SBI_PROB", 0.0) <= 0: print("WARNING: USE_MASK_HEAD without SBI_PROB>0 -> the mask head only ever sees all-zero targets", flush=True)

def nov_banner(cfg):
    ps = cfg.get("SBI_PROB", 0.0) + cfg.get("FDSBI_PROB", 0.0); fake_share = (1 - cfg["REAL_FRAC"]) + cfg["REAL_FRAC"] * ps
    log(f"NOVELTY | SBI_PROB={cfg.get('SBI_PROB', 0.0)} FDSBI_PROB={cfg.get('FDSBI_PROB', 0.0)} USE_MASK_HEAD={cfg.get('USE_MASK_HEAD', False)} LAMBDA_MASK={cfg.get('LAMBDA_MASK', 0.0)} | "
        f"pseudo-fake share of REAL draws={ps:.2f} -> fake-clip share per epoch ~{fake_share:.0%} (REAL_FRAC={cfg['REAL_FRAC']})")

# ---------------------------------- C2: FD-SBI ----------------------------------
_NOV_GRID = {}
def _nov_fd_grid(N):
    if N not in _NOV_GRID:
        f = np.fft.fftshift(np.fft.fftfreq(N)).astype(np.float32) * 2.0                      # normalised frequency, Nyquist = 1, DC at index N//2
        FY, FX = np.meshgrid(f, f, indexing="ij"); _NOV_GRID[N] = (FX, FY, np.sqrt(FX ** 2 + FY ** 2))
    return _NOV_GRID[N]

def nov_fd_gain(q, N):
    """Log-gain field (even-symmetric about DC, so the result stays a real image). None for the 'replica' mode (data dependent)."""
    FX, FY, R = _nov_fd_grid(N)
    if q["mode"] in ("atten", "boost"): lg = (-1.0 if q["mode"] == "atten" else 1.0) * q["s"] * np.exp(-0.5 * ((R - q["r0"]) / q["rw"]) ** 2)      # radial band
    elif q["mode"] == "peaks":                                                                                                                      # symmetric spectral peak pairs (periodic, GAN-upsampling-like)
        lg = np.zeros_like(R)
        for j in range(q["k"]):
            fx, fy = q["pr"][j] * math.cos(q["pa"][j]), q["pr"][j] * math.sin(q["pa"][j])
            for sg in (1.0, -1.0): lg = lg + q["s"] * np.exp(-0.5 * (((FX - sg * fx) ** 2 + (FY - sg * fy) ** 2) / q["psig"] ** 2))
    else: return None
    return np.exp(lg * (R > 0.04)).astype(np.float32)                                         # DC / lowest frequencies untouched

def nov_fd_blend(im, q, pad=32):
    """Frequency-domain self-blend: |X'| = spectral edit of |X|, phase(X) kept. Reflect-padded (pad px) to avoid wrap-around ringing; result cropped back to the original size."""
    a = np.asarray(im, np.float32); S = a.shape[0]; N = S + 2 * pad
    ap = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    X = np.fft.fftshift(np.fft.fft2(ap, axes=(0, 1)), axes=(0, 1)); mag = np.abs(X); ph = X / np.maximum(mag, 1e-6)
    if q["mode"] == "replica":                                                                # aliasing-like spectral replica: shifted copy of the magnitude added to the high band (capped at 3x local magnitude)
        R = _nov_fd_grid(N)[2]; rep = 0.5 * (np.roll(mag, (q["shy"], q["shx"]), (0, 1)) + np.roll(mag, (-q["shy"], -q["shx"]), (0, 1)))
        mag2 = mag + q["beta"] * np.minimum(rep, 3.0 * mag) * (R > 0.3)[..., None]
    else:
        g = q.get("_g")
        if g is None: g = q["_g"] = nov_fd_gain(q, N)
        mag2 = mag * g[..., None]
    y = np.fft.ifft2(np.fft.ifftshift(mag2 * ph, axes=(0, 1)), axes=(0, 1)).real[pad:pad + S, pad:pad + S]
    return Image.fromarray(np.clip(np.rint(y), 0, 255).astype(np.uint8))

# ---------------------------------- C1: mask head loss ----------------------------------
def nov_aux_dev(aux, dev): return {k: v.to(dev, non_blocking=True) for k, v in aux.items()}

def nov_mask_loss(out, aux):
    """BCE on every clip with a known target (REAL clips: all-zero target; SBI clips: blurred blend mask) + Dice on the SBI clips only.
    FF++ fakes and FD-SBI clips have NO spatial mask target -> excluded (mvalid = 0)."""
    ml = out["mask_logits"].float(); B, T, h, w = ml.shape; m = aux["mask"].float(); tgt = m.unsqueeze(1).expand(B, T, h, w); valid = aux["mvalid"].float()
    z = ml.sum() * 0.0; nv = valid.sum()
    if nv < 1: return z, z, z
    bce = (F.binary_cross_entropy_with_logits(ml, tgt, reduction="none").mean((1, 2, 3)) * valid).sum() / nv
    pos = (m.sum((1, 2)) > 0).float() * valid; npos = pos.sum()
    if npos >= 1:
        p = torch.sigmoid(ml); inter = (p * tgt).sum((2, 3)); den = p.sum((2, 3)) + tgt.sum((2, 3))
        dice = ((1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean(1) * pos).sum() / npos
    else: dice = z
    return bce + dice, bce.detach(), dice.detach()

def compute_loss(out, y, cfg, gscale=1.0, kscale=1.0, aux=None):
    L = _compute_loss_orig(out, y, cfg, gscale, kscale); z = torch.zeros((), device=y.device); L["mask"] = z
    if "mask_logits" in out and aux is not None and cfg.get("LAMBDA_MASK", 0.0) > 0:
        lm, _, _ = nov_mask_loss(out, aux); L["mask"] = lm; L["total"] = L["total"] + cfg["LAMBDA_MASK"] * lm
    return L

def nov_epoch_log(epoch, acc, cfg):
    if cfg.get("USE_MASK_HEAD"): n_ = max(1.0, acc["n"].item()); log(f"Epoch {epoch+1}: TRAIN boundary-mask loss (BCE+Dice) = {acc['mask'].item() / n_:.4f}")

# ---------------------------------- C3: conformal thresholds ----------------------------------
def conformal_threshold(cal_scores, alpha):
    """Split-conformal (1-alpha) quantile: k = ceil((n+1)(1-alpha))-th smallest calibration score (inf if k > n). Used with the STRICT rule 'flag when score > tau'."""
    s = np.sort(np.asarray(cal_scores, dtype=float)); n = len(s); k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if (k > n or n == 0) else float(s[k - 1])

def _wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - a) / d, (c + a) / d

def _nov_op_row(name, method, alpha, y, p, pred, thr):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int); nr, nf = int((y == 0).sum()), int((y == 1).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tp = int(((pred == 1) & (y == 1)).sum()); fpr = fp / max(1, nr); tpr = tp / max(1, nf); lo, hi = _wilson(fp, nr)
    prec = tp / max(1, tp + fp); f1 = 2 * prec * tpr / max(1e-12, prec + tpr)
    return dict(dataset=name, method=method, alpha=alpha, threshold=thr, n_real=nr, n_fake=nf, fpr=fpr, fpr_ci_low=lo, fpr_ci_high=hi, fpr_minus_alpha=(fpr - alpha) if alpha is not None else float("nan"),
                tpr=tpr, bal_acc=0.5 * (tpr + 1 - fpr), f1=f1, acc=(tp + nr - fp) / max(1, nr + nf))

def nov_conformal_report(cfg, R, vv, res, thr_v):
    try: _nov_conformal_report(cfg, R, vv, res, thr_v)
    except Exception as e:
        import traceback; traceback.print_exc(); log(f"!! conformal report FAILED (the main evaluation is unaffected): {type(e).__name__}: {e}")

def _nov_conformal_report(cfg, R, vv, res, thr_v):
    alphas = [float(a) for a in cfg.get("CONFORMAL_ALPHAS", (0.01, 0.02, 0.05, 0.10))]
    yv = vv["label"].values.astype(int); pv = vv["p_video"].values.astype(float); cal_r, cal_f = pv[yv == 0], pv[yv == 1]
    log(f"Conformal calibration set = FF++ VAL videos: {len(cal_r)} real, {len(cal_f)} fake | alphas={alphas}")
    thr_rows, op_rows, set_rows = [], [], []
    for a in alphas: thr_rows.append(dict(alpha=a, tau_real=conformal_threshold(cal_r, a), q_fake=conformal_threshold(1.0 - cal_f, a), n_cal_real=len(cal_r), n_cal_fake=len(cal_f), youden_threshold=thr_v))
    for key, (name, fdf, vdf, thr_f, tv) in res.items():
        y = vdf["true_label"].values.astype(int); p = vdf["predicted_probability"].values.astype(float)
        op_rows.append(_nov_op_row(name, "youden_val", None, y, p, p >= tv, tv))
        for a, tr in zip(alphas, thr_rows):
            op_rows.append(_nov_op_row(name, "conformal_real", a, y, p, p > tr["tau_real"], tr["tau_real"]))
            inR = p <= tr["tau_real"]; inF = (1.0 - p) <= tr["q_fake"]; single = inR ^ inF; both = inR & inF; empty = ~inR & ~inF
            sel_acc = float(((inF & ~inR & (y == 1)) | (inR & ~inF & (y == 0))).sum() / max(1, single.sum()))
            set_rows.append(dict(dataset=name, alpha=a, target_coverage=1 - a, cov_real=float(inR[y == 0].mean()) if (y == 0).any() else float("nan"), cov_fake=float(inF[y == 1].mean()) if (y == 1).any() else float("nan"),
                                 frac_singleton=float(single.mean()), frac_both_uncertain=float(both.mean()), frac_empty=float(empty.mean()), selective_acc_on_singletons=sel_acc, n=len(y)))
    pd.DataFrame(thr_rows).to_csv(os.path.join(R["metrics"], "conformal_thresholds.csv"), index=False)
    ops = pd.DataFrame(op_rows); ops.to_csv(os.path.join(R["metrics"], "conformal_operating_points.csv"), index=False)
    sets = pd.DataFrame(set_rows); sets.to_csv(os.path.join(R["metrics"], "conformal_sets.csv"), index=False)
    ff = lambda v: f"{v:.3f}"
    print("\n" + "=" * 120 + "\nCONFORMAL (C3) — real-class threshold from FF++ val (flag fake if p > tau) vs the Youden threshold | FPR on REAL videos should be <= alpha ONLY where val/test are exchangeable (FF++ test)\n" + "=" * 120)
    print(ops[["dataset", "method", "alpha", "threshold", "fpr", "fpr_ci_low", "fpr_ci_high", "tpr", "bal_acc", "f1"]].to_string(index=False, float_format=ff))
    print("\nCLASS-CONDITIONAL CONFORMAL SETS (coverage target = 1-alpha per class; 'uncertain' = both labels kept, 'empty' = neither)"); print(sets.to_string(index=False, float_format=ff))
    try:
        fig, ax = plt.subplots(figsize=(5.2, 4.4)); grid = np.linspace(0.01, 0.5, 50)
        for key, (name, fdf, vdf, thr_f, tv) in res.items():
            y = vdf["true_label"].values.astype(int); p = vdf["predicted_probability"].values.astype(float); pr = p[y == 0]
            if len(pr): ax.plot(grid, [float((pr > conformal_threshold(cal_r, a)).mean()) for a in grid], lw=1.8, label=name)
        ax.plot(grid, grid, "k--", lw=0.9, label="nominal (FPR = alpha)"); ax.set_xlabel("target alpha"); ax.set_ylabel("empirical FPR on real videos"); ax.set_title("Conformal real-class threshold: nominal vs empirical FPR")
        ax.legend(fontsize=7); savefig(fig, os.path.join(R["figs"], "conformal_fpr_vs_alpha"))
    except Exception as e: log(f"conformal figure skipped: {type(e).__name__}: {e}")


# ---------------------------------- self-test (CPU, synthetic data; NOV_SELFTEST=1 quick, =2 adds a short learning check) ----------------------------------
def nov_selftest(level=1):
    import tempfile
    log("NOV SELFTEST: start (CPU, synthetic frames, no dataset needed)"); rs = np.random.default_rng(0); tmp = tempfile.mkdtemp(); dev = torch.device("cpu"); NORM["type"] = "gn"
    def mkvid(name, label, n=16):
        d = os.path.join(tmp, name); os.makedirs(d); yy, xx = np.mgrid[0:256, 0:256] / 256.0
        for j in range(n):
            g = 128 + 55 * np.sin(6 * xx + 0.1 * j) * np.cos(5 * yy) + rs.normal(0, 5, (256, 256)); Image.fromarray(np.stack([g, g * 0.9, g * 0.8], -1).clip(0, 255).astype(np.uint8)).save(os.path.join(d, f"{j:04d}.jpg"), quality=95)
        return dict(video_id=name, vname=name, label=label, generator="real" if label == 0 else "fake", dir=d, files=sorted(os.listdir(d)), n=n)
    reals = [mkvid(f"r{i}", 0) for i in range(2)]; fakes = [mkvid(f"f{i}", 1) for i in range(2)]
    base = dict(CONFIG, NUM_FRAMES=4, TRAIN_WINDOW=12, TEMPORAL_STRIDE=2, SEED=7, AUG_RAMP_EPOCHS=0)
    # 1) dataset: kinds / labels / mask targets / determinism
    for sp, fp in ((0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        cfg = dict(base, SBI_PROB=sp, FDSBI_PROB=fp); ds = ClipDataset(reals + fakes, cfg, True); ds.set_epoch(1); seen = defaultdict(int)
        for pos in range(24):
            for i in range(len(ds)):
                it = ds[(pos, i)]; assert len(it) == 4 and it[0].shape == (4, 224, 224, 3) and it[0].dtype == torch.uint8; x, y, _, a = it; k = int(a["kind"]); seen[k] += 1
                v = ds.v[ds.items[i][0]]
                if v["label"] == 1: assert k == 1 and int(y) == 1 and float(a["mvalid"]) == 0
                elif k == 0: assert int(y) == 0 and float(a["mvalid"]) == 1 and float(a["mask"].sum()) == 0
                elif k == 2: assert int(y) == 1 and float(a["mvalid"]) == 1 and a["mask"].shape == (28, 28) and 0 <= float(a["mask"].min()) and float(a["mask"].max()) <= 1.0 and float(a["mask"].sum()) > 20
                elif k == 3: assert int(y) == 1 and float(a["mvalid"]) == 0 and float(a["mask"].sum()) == 0
                assert torch.equal(it[0], ds[(pos, i)][0]), "train item not deterministic"
        log(f"  dataset SBI={sp} FDSBI={fp}: kinds seen {dict(seen)} OK")
        if (sp, fp) == (0.5, 0.5): assert seen[2] > 0 and seen[3] > 0
        if (sp, fp) == (0.0, 0.0): assert set(seen) <= {0, 1}
    ev = ClipDataset(reals, base, False, 2); assert len(ev[0]) == 3, "EVAL items must stay 3-tuples"
    # 2) SBI geometry refactor: pixels outside the mask are untouched
    ds = ClipDataset(reals, dict(base, SBI_PROB=1.0), True); im = Image.fromarray(rs.integers(0, 255, (224, 224, 3), dtype=np.uint8)); q = ds._make_sbi(np.random.default_rng(3))
    o = np.asarray(ds._self_blend(im, q)).astype(int); a0 = np.asarray(im).astype(int); m = ds._sbi_mask(q); far = m * q["alpha"] < 1e-6
    assert far.sum() > 1000 and np.abs(o - a0)[far].max() == 0 and np.abs(o - a0)[m > 0.9].mean() > 5; log("  SBI mask == blend support (outside untouched, inside changed) OK")
    # 3) FD-SBI: every mode changes the image modestly, keeps phase, stays real/uint8, deterministic
    smooth = ds.__class__._prep  # noqa (keep reference alive)
    imf = Image.open(os.path.join(reals[0]["dir"], "0000.jpg")).resize((224, 224)); a0 = np.asarray(imf).astype(np.float32)
    for mode in ("atten", "boost", "peaks", "replica"):
        for sd in range(4):
            q = ds._make_fdsbi(np.random.default_rng(sd)); q["mode"] = mode; o = np.asarray(nov_fd_blend(imf, q)); assert o.shape == (224, 224, 3) and o.dtype == np.uint8
            o2 = np.asarray(nov_fd_blend(imf, dict(q))); assert np.array_equal(o, o2)
            mse = float(((o.astype(np.float32) - a0) ** 2).mean()); psnr = 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse); assert psnr > 22, f"{mode} too destructive (PSNR {psnr:.1f})"
        d = np.abs(np.fft.fft2(o[..., 1].astype(np.float32))) / np.maximum(np.abs(np.fft.fft2(a0[..., 1])), 1e-3); log(f"  FD-SBI mode={mode}: last PSNR {psnr:.1f} dB OK")
    # 4) model: mask head on/off, shapes, gradients, state-dict compatibility
    mk = lambda **kw: dict(base, BACKBONES=("resnet18",), PRETRAINED=False, USE_AMP=False, EMBED_DIM=64, GENCONVIT_VARIANT="ed", NUM_FRAMES=2, TEMP_LAYERS=1, DROPOUT=0.0, **kw)
    cfg0 = mk(USE_MASK_HEAD=False); m0 = StudentModel(cfg0); assert m0.mhead is None and not any(k.startswith("mhead") for k in m0.state_dict()), "mask head leaked into the no-head model"
    cfg1 = mk(USE_MASK_HEAD=True, SBI_PROB=0.5, FDSBI_PROB=0.5); m1 = StudentModel(cfg1); m1.train()
    ds = ClipDataset(reals + fakes, dict(cfg1, SBI_PROB=0.7, FDSBI_PROB=0.3), True); ds.set_epoch(0)
    dl = DataLoader(ds, batch_size=4, sampler=[(k, k % len(ds)) for k in range(4)], num_workers=0); x, y, _, aux = next(iter(dl)); assert aux["mask"].shape == (4, 28, 28) and aux["mvalid"].shape == (4,)
    x01 = x.permute(0, 1, 4, 2, 3).float() / 255.0; out = m1(x01); assert out["mask_logits"].shape == (4, 2, 28, 28), out["mask_logits"].shape
    L = compute_loss(out, y, cfg1, aux=aux); assert torch.isfinite(L["total"]) and "mask" in L; L["total"].backward()
    gm = sum(float(p.grad.abs().sum()) for p in m1.mhead.parameters() if p.grad is not None); ga = sum(float(p.grad.abs().sum()) for p in m1.aenc.parameters() if p.grad is not None)
    assert gm > 0 and ga > 0, "mask loss gradient does not reach the head / artifact encoder"
    L0 = compute_loss(m0(x01), y, cfg0); assert float(L0["mask"]) == 0.0; log(f"  model: mask_logits {tuple(out['mask_logits'].shape)}, mask loss {float(L['mask']):.3f}, grad head {gm:.2e} / encoder {ga:.2e}; no-head model unchanged OK")
    # loss with every clip invalid -> zero, no crash
    a2 = dict(aux, mvalid=torch.zeros(4)); L2 = compute_loss(m1(x01), y, cfg1, aux=a2); assert float(L2["mask"]) == 0.0
    if level >= 2:
        opt = torch.optim.Adam(list(m1.aenc.parameters()) + list(m1.mhead.parameters()), lr=2e-3); first = None
        for st in range(25):
            o = m1(x01); lm = nov_mask_loss(o, aux)[0]; opt.zero_grad(); lm.backward(); opt.step(); first = float(lm) if first is None else first
        log(f"  mask head learning check: loss {first:.3f} -> {float(lm):.3f}"); assert float(lm) < 0.8 * first, "mask head does not learn"
    # 5) conformal: finite-sample marginal validity under exchangeability + edge cases + end-to-end report
    r = np.random.default_rng(1)
    for a in (0.01, 0.05, 0.10):
        fprs = []
        for _ in range(300): cal = r.beta(1, 8, 140); te = r.beta(1, 8, 4000); fprs.append(float((te > conformal_threshold(cal, a)).mean()))
        mf = float(np.mean(fprs)); assert a - 0.012 <= mf <= a + 0.004, f"conformal FPR {mf:.4f} vs alpha {a}"
    assert conformal_threshold(np.arange(10) / 10, 0.01) == float("inf") and conformal_threshold([], 0.1) == float("inf")
    assert conformal_threshold(np.arange(1, 101) / 100, 0.05) == 0.96     # n=100: k=ceil(101*0.95)=96 -> 0.96
    R_ = dict(metrics=os.path.join(tmp, "m"), figs=os.path.join(tmp, "f")); os.makedirs(R_["metrics"]); os.makedirs(R_["figs"]); setup_mpl()
    vv = pd.DataFrame(dict(label=[0] * 140 + [1] * 560, p_video=np.r_[r.beta(1, 8, 140), r.beta(8, 1, 560)]))
    mkv = lambda n0, n1, s: pd.DataFrame(dict(true_label=[0] * n0 + [1] * n1, predicted_probability=np.r_[r.beta(1, 8 - s, n0), r.beta(8 - s, 1, n1)]))
    res = {"ffpp": ("FF++", None, mkv(140, 140, 0), 0.5, 0.5), "celebdf": ("Celeb-DF", None, mkv(100, 200, 4), 0.5, 0.5)}
    _nov_conformal_report(dict(CONFIG), R_, vv, res, 0.5); ops = pd.read_csv(os.path.join(R_["metrics"], "conformal_operating_points.csv")); sets = pd.read_csv(os.path.join(R_["metrics"], "conformal_sets.csv"))
    assert len(ops) == 2 * (1 + 4) and len(sets) == 2 * 4 and os.path.exists(os.path.join(R_["figs"], "conformal_fpr_vs_alpha.png")) and os.path.exists(os.path.join(R_["metrics"], "conformal_thresholds.csv"))
    assert abs((sets.frac_singleton + sets.frac_both_uncertain + sets.frac_empty) - 1).max() < 1e-9
    shutil.rmtree(tmp, ignore_errors=True); log("NOV SELFTEST PASSED (dataset kinds/masks, SBI refactor, FD-SBI modes, mask head on/off + gradients, conformal validity + report)")


class TeacherAsModel(nn.Module):
@@END
NOV_DSL_EOF
cat > $W/nov/nov_apply.py <<'NOV_PY_EOF'
# Anchor-checked patcher: reads the Cell-1 script, applies every PATCH from the .dsl file, writes a NEW file (Cell-1 file is never modified).
# Every anchor must occur EXACTLY ONCE in the source, otherwise nothing is written and the cell stops (no silent half-patched script).
import sys, re
src_path, dst_path, dsl_path = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(src_path, encoding="utf-8").read()
if "NOVELTY PATCH (C1 mask head" in src: sys.exit("source already patched - give the ORIGINAL Cell-1 script")
dsl = open(dsl_path, encoding="utf-8").read()
blocks = re.findall(r"^@@PATCH (.+?)\n(.*?)\n@@WITH\n(.*?)\n@@END$", dsl, flags=re.S | re.M)
if not blocks: sys.exit("no patches found in dsl")
out = src; bad = []
for name, old, new in blocks:
    n = out.count(old)
    if n != 1: bad.append(f"  [{name}] anchor found {n}x (need exactly 1): {old[:90]!r}"); continue
    out = out.replace(old, new)
if bad: sys.exit("PATCH FAILED - the Cell-1 script differs from the one this patch was written for:\n" + "\n".join(bad))
compile(out, dst_path, "exec")
open(dst_path, "w", encoding="utf-8").write(out)
print(f"novelty patch applied: {len(blocks)} edits -> {dst_path} ({len(out.splitlines())} lines)")
NOV_PY_EOF
cat > $W/nov/nov_agg.py <<'NOV_AGG_EOF'
# usage: python nov_agg.py <out_csv>   -> prints + saves the video-level summary of every finished *_eval run in /kaggle/working/output
import glob, os, sys, pandas as pd
O = "/kaggle/working/output"; rows = []; out_csv = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/novelty_summary_this_notebook.csv"
for d in sorted(glob.glob(f"{O}/*_eval")):
    n = os.path.basename(d)[:-5]; f = f"{d}/metrics/final_metrics_summary.csv"; c = f"{d}/metrics/conformal_operating_points.csv"
    if not os.path.exists(f): continue
    s = pd.read_csv(f); s = s[(s.level == "video")]
    for _, r in s.iterrows(): rows.append(dict(run=n, dataset=r.dataset, auc=r.auc, eer=r.eer, bal_acc_youden=r.bal_acc, fpr_youden=1 - r.specificity, tpr_youden=r.recall))
    if os.path.exists(c):
        k = pd.read_csv(c); k = k[(k.method == "conformal_real") & (k.alpha.round(4) == 0.05)]
        for _, r in k.iterrows():
            for x in rows:
                if x["run"] == n and x["dataset"] == r.dataset: x.update(fpr_conf05=r.fpr, tpr_conf05=r.tpr, bal_acc_conf05=r.bal_acc)
if not rows: print("no finished evaluations yet")
else:
    df = pd.DataFrame(rows); pd.set_option("display.width", 220); print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(out_csv, index=False); print("saved", out_csv)
NOV_AGG_EOF
cat > $W/nov/nov_zip.py <<'NOV_ZIP_EOF'
# usage: python nov_zip.py <run_name>  -> /kaggle/working/<run_name>_results.zip with ALL metrics / figures / predictions / logs / configs of the training run AND its evaluation (NO .pth checkpoints)
import os, sys, zipfile, glob
run = sys.argv[1]; W, O = "/kaggle/working", "/kaggle/working/output"; dst = f"{W}/{run}_results.zip"; tmp = dst + ".tmp"; n = 0
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    for d in (run, run + "_eval"):
        for root, _, files in os.walk(f"{O}/{d}"):
            for f in files:
                if f.endswith((".pth", ".tmp")): continue
                p = os.path.join(root, f); z.write(p, os.path.join("output", os.path.relpath(p, O))); n += 1
    extra = glob.glob(f"{W}/log_{run}*.txt") + [f"{W}/{run}_STATUS.txt", f"{W}/novelty_summary_{run}.csv", f"{W}/nov/selftest.txt"]
    for p in extra:
        if os.path.exists(p): z.write(p, os.path.join("top_level", os.path.basename(p))); n += 1
os.replace(tmp, dst); print(f"ZIP written: {dst} | {n} files | {os.path.getsize(dst)/2**20:.1f} MB")
NOV_ZIP_EOF

# 1) patch Cell-1 script (anchor-checked; aborts loudly if any anchor is missing) -> genconvit_nov.py
python $W/nov/nov_apply.py $W/genconvit_tfkd.py $SCRIPT $W/nov/nov_patch.dsl || exit 1
# 2) fast CPU self-test of every new component BEFORE spending hours of GPU time
NOV_SELFTEST=1 python $SCRIPT > $W/nov/selftest.txt 2>&1; RC=$?; tail -n 4 $W/nov/selftest.txt
[ $RC -eq 0 ] || { echo "NOV SELFTEST FAILED (see $W/nov/selftest.txt)"; exit 1; }
# 3) check/extract faces once
GC_EXTRACT_ONLY=1 python $SCRIPT || exit 1

train_done() { [ -f $CK/best_auc.pth ] && [ ! -f $CK/last.pth ]; }

# ------------------------------------------------ TRAINING (this seed only, GPU 0, 4 workers) ------------------------------------------------
if train_done; then echo "skip training $RUN (already finished)"; else
  CUDA_VISIBLE_DEVICES=0 GC_OVERRIDES="{\"MODE\":\"train\",\"USE_KD\":false,\"NUM_WORKERS\":4,\"NUM_WORKERS_EVAL\":4,\"EXP_NAME\":\"$RUN\",\"SEED\":$SEED,\"REAL_FRAC\":0.5,\"SBI_PROB\":0.3,\"FDSBI_PROB\":0.0,\"USE_MASK_HEAD\":true,\"LAMBDA_MASK\":0.5,\"NUM_EPOCHS\":$NUM_EPOCHS,\"TRAIN_CLIPS_PER_EPOCH\":$CLIPS,\"MAX_SESSION_MINUTES\":$SOFT_MIN}" \
    timeout -k 60 ${TRAIN_HARD_MIN}m python $SCRIPT > $W/log_$RUN.txt 2>&1
  echo "training exit code: $?   (124 = hard timeout)"
fi
tail -n 6 $W/log_$RUN.txt 2>/dev/null

# keep only best_auc.pth once all epochs are done (an unfinished run keeps last.pth so it stays resumable)
E=$(printf "%03d" $NUM_EPOCHS)
[ -f $CK/session_end_epoch$E.pth ] && rm -f $CK/last.pth $CK/ckpt_epoch*.pth $CK/session_end_epoch$E.pth

# status file (goes into the ZIP): was the schedule completed?
{ echo "run=$RUN seed=$SEED NUM_EPOCHS=$NUM_EPOCHS TRAIN_CLIPS_PER_EPOCH=$CLIPS";
  if train_done; then echo "TRAINING: FINISHED all $NUM_EPOCHS epochs";
  else echo "TRAINING: INCOMPLETE (time budget / error) -> evaluation uses best_auc.pth so far; NOT comparable to a completed run"; fi
  grep -E "Stopping training|Current epoch|Best validation AUC" $W/log_$RUN.txt 2>/dev/null | tail -n 3; } > $W/${RUN}_STATUS.txt
cat $W/${RUN}_STATUS.txt
python $W/nov/nov_zip.py $RUN      # safety ZIP right after training (overwritten after evaluation)

# ------------------------------------------------ EVALUATION ------------------------------------------------
ELAPSED=$(( ($(date +%s) - T0) / 60 )); echo "elapsed since this cell started: ${ELAPSED} min"
if [ ! -f $CK/best_auc.pth ]; then
  echo "!! no best_auc.pth for $RUN (training produced no validated epoch) -> evaluation impossible; see log_$RUN.txt"
elif [ -f $OUT/${RUN}_eval/metrics/final_metrics_summary.csv ]; then
  echo "skip evaluation $RUN (already done)"
elif [ $(( ELAPSED + EVAL_HARD_MIN )) -gt $SESSION_LIMIT_MIN ]; then
  echo "!! Not enough session time left for evaluation (${ELAPSED} + ${EVAL_HARD_MIN} > ${SESSION_LIMIT_MIN}) -> SKIPPED. Safety ZIP of training is written."
else
  # SEED left at its default 42 on purpose (same Diffusion subset as your earlier runs)
  CUDA_VISIBLE_DEVICES=0 GC_OVERRIDES="{\"MODE\":\"evaluate\",\"NUM_WORKERS\":4,\"NUM_WORKERS_EVAL\":4,\"EXP_NAME\":\"${RUN}_eval\",\"BEST_CHECKPOINT\":\"$CK/best_auc.pth\"}" \
    timeout -k 60 ${EVAL_HARD_MIN}m python $SCRIPT > $W/log_${RUN}_eval.txt 2>&1
  echo "evaluation exit code: $?   (124 = hard timeout)"
  grep -E "Thresholds|video AUC|EVALUATION COMPLETE|Error|Traceback" $W/log_${RUN}_eval.txt | tail -n 10
  python $W/nov/nov_agg.py $W/novelty_summary_$RUN.csv
fi

# ------------------------------------------------ FINAL ZIP (always written, even if evaluation was skipped / killed) ------------------------------------------------
python $W/nov/nov_zip.py $RUN
ls -la $W/${RUN}_results.zip
echo "DONE: download ${RUN}_results.zip from the Output tab"