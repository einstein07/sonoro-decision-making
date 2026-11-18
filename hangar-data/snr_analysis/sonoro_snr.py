#!/usr/bin/env python3
"""Signal-to-noise ratio of the sound input of the Sonoro spin ring attractor.

Works backwards from what the ring attractor actually received.  Every control
cycle of ring_circle_target.py logs the 120-value external field hext that was
fed to the ring.  hext is the max-aggregated DMAS-CF spatial response of the
listening window, min-max normalised to [0, 1] and multiplied by 0.19, so each
row contains one lobe per speaker and the taller lobe is exactly 0.19.

Per cycle n we read the two lobe heights (in units of the taller lobe)

    q_L[n], q_R[n]        (left speaker at positive ring angles)

and form the sample of the quality difference used in the ICRA 2027 paper,

    o[n] = q_L[n] - q_R[n],          o[n] = Dq(pose[n]) + noise,

whose mean Dq is the signal and whose scatter sigma_o about that mean is the
noise.  In the paper each target's sample carries independent noise eta, so
sigma_o = sqrt(2) eta and the per-sample signal-to-noise ratio is

    D = Dq / eta = sqrt(2) Dq / sigma_o,      p = Phi(D / sqrt(2)).

The robot moves, so Dq depends on the pose.  Two routes separate the
pose-dependent mean from the noise:

  Route A  regression of ln(q_R/q_L) on the measured geometry (Qualisys);
           the noise is the residual.
  Route B  successive differences between consecutive cycles (ringStats only);
           the robot moves 13 mm per cycle, so the mean barely changes and
           half the variance of the differences is the noise variance.

Route A can only overstate the noise (it counts what its pose model misses),
Route B can only understate it (it removes anything slow).  The script also
replays the ring on the logged inputs (bit-exact check of the log), checks the
clocks, and replays every run with other random seeds.

Usage, from the repository root:
    python snr_analysis/sonoro_snr.py [--root <repo>] [--out <dir>] [--boot 2000] [--seeds 200]
    python snr_analysis/make_figures.py

Needs numpy and pandas (scipy only for the Shapiro-Wilk test, matplotlib for
the figures).  The run list, the speaker levels and the reported outcomes of
experiment A1 are set just below.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Experiment description (16 Nov 2026, experiment A1)
# ----------------------------------------------------------------------------
DATE = "16_11"
RUNS = ["01", "03", "04", "05", "06", "07", "08", "09", "10", "11"]      # A1_02 excluded
RIGHT_LEVEL = {"01": 0.7, "08": 0.7, "09": 0.7, "10": 0.7, "11": 0.7,     # left level = 1.0
               "03": 0.6, "04": 0.6, "05": 0.6, "06": 0.6, "07": 0.6}
REPORTED_CHOICE = {"01": "R", "08": "R", "09": "R", "10": "R", "11": "L",
                   "03": "L", "04": "L", "05": "L", "06": "L", "07": "L"}
# the sound controller is restarted when the levels change: three sessions
SESSION = {"01": 1, "03": 2, "04": 2, "05": 2, "06": 2, "07": 2,
           "08": 3, "09": 3, "10": 3, "11": 3}

# controller constants (ring_circle_target.py)
N = 120
THETA = -180.0 + 3.0 * np.arange(N)       # ring angle of unit i, deg; positive = robot's left
H_SCALE = 0.19                            # DOA_NORMALIZING_FACTOR
RING_V, RING_HB, RING_BETA, RING_UPDATES, RING_SEED = 0.4, 0.045, 400.0, 1200, 300
ACQ_S = 3.0                               # listening window per cycle, s

START_CYCLES = 15                         # "start" window: cycles 0..14
COMMIT_CYCLES = 8                         # "pre-commitment" window: the 8 cycles before the turn
TURN_DEG = 10.0                           # a turn larger than this, kept, is the commitment
SETTLE_CYCLES = 15                        # start-up cycles ignored when replaying with other seeds
BLOCK_CYCLES = 7                          # block length of the along-approach curves
ALPHA_REF, ALPHA_SCALE = 70.0, 25.0       # centring of the lobe separation in the regression


def wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_ring(root, run):
    d = pd.read_csv(os.path.join(root, "ringStats", f"{DATE}_A1_{run}.csv"))
    return dict(spins=d[[f"spin_{i}" for i in range(N)]].to_numpy(int),
                hext=d[[f"hext_{i}" for i in range(N)]].to_numpy(float),
                bump=d["bump_angle"].to_numpy(float),
                t=pd.to_datetime(d["timestamp"]), n=len(d))


def load_traj(root, run):
    f = os.path.join(root, "trajectories", f"{DATE}_A1_{run}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    if len(d) == 0:
        return None
    d["t"] = pd.to_datetime(d["timestamp"])
    return d


# ----------------------------------------------------------------------------
# Lobes of one input profile
# ----------------------------------------------------------------------------
def _refine(q, i):
    """Three-point parabolic refinement of a local maximum: (angle, height)."""
    a, b, c = q[(i - 1) % N], q[i], q[(i + 1) % N]
    den = a - 2.0 * b + c
    if den >= 0:
        return float(THETA[i]), float(b)
    off = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return float(wrap(THETA[i] + 3.0 * off)), float(b - 0.25 * (a - c) * off)


def lobes(h, floor=0.02):
    """Local maxima of a profile, strongest first, as [(angle_deg, height)],
    heights in units of the profile maximum."""
    q = np.asarray(h, float) / H_SCALE
    idx = np.where((q > np.roll(q, 1)) & (q >= np.roll(q, -1)) & (q > floor))[0]
    idx = idx[np.argsort(q[idx])[::-1]]
    return [_refine(q, i) for i in idx]


# ----------------------------------------------------------------------------
# The ring attractor of ring_circle_target.py, for the bit-exact replay
# ----------------------------------------------------------------------------
class SpinRing:
    """Faithful re-implementation of RingAttractor / compute_delta_H / run_ring_attractor."""

    def __init__(self, seed=RING_SEED):
        self.rs = np.random.RandomState(seed)             # np.random.seed(300) in main()
        self.th = np.linspace(-np.pi, np.pi, N, endpoint=False)
        self.spins = self.rs.choice([1, 0], size=N)
        d = np.abs((self.th[None, :] - self.th[:, None] + np.pi) % (2 * np.pi) - np.pi)
        self.J = np.cos(np.pi * (d / np.pi) ** RING_V)
        np.fill_diagonal(self.J, 0.0)

    def step(self, h):
        for _ in range(RING_UPDATES):
            i = self.rs.randint(0, N)
            s = self.spins[i]
            flip = np.array([s, 1 - s])
            ham = -(np.dot(self.J[i], self.spins) * flip / (N - 1) + h[i] * flip - RING_HB * flip)
            dh = ham[1] - ham[0]
            if dh < 0:
                self.spins[i] = 1 - s
            elif self.rs.rand() < np.exp(-RING_BETA * dh):
                self.spins[i] = 1 - s
        on = self.spins == 1
        return math.degrees(np.angle(np.sum(np.exp(1j * self.th[on])))) if on.any() else 0.0


def replay_check(ring):
    """Replay the ring on the logged inputs; count spins that differ from the log."""
    sim = SpinRing()
    wrong, dbump = 0, 0.0
    for k in range(ring["n"]):
        b = sim.step(ring["hext"][k])
        wrong += int(np.sum(sim.spins != ring["spins"][k]))
        dbump = max(dbump, abs(b - ring["bump"][k]))
    return wrong, dbump


def replay_seeds(hext, n_cycles, seeds):
    """Replay the first n_cycles of a run with other random seeds (vectorised over
    seeds).  Returns the spin states after every cycle, shape (seeds, cycles, N)."""
    S = len(seeds)
    th = np.linspace(-np.pi, np.pi, N, endpoint=False)
    d = np.abs((th[None, :] - th[:, None] + np.pi) % (2 * np.pi) - np.pi)
    J = np.cos(np.pi * (d / np.pi) ** RING_V)
    np.fill_diagonal(J, 0.0)
    rng = np.random.default_rng(seeds[0] * 7919 + 13)
    spins = rng.integers(0, 2, size=(S, N))
    out = np.zeros((S, n_cycles, N), dtype=np.int8)
    ar = np.arange(S)
    for k in range(n_cycles):
        h = hext[k]
        idx = rng.integers(0, N, size=(RING_UPDATES, S))
        u = rng.random(size=(RING_UPDATES, S))
        for m in range(RING_UPDATES):
            i = idx[m]
            s = spins[ar, i]
            field = np.einsum("sj,sj->s", J[i], spins) / (N - 1) + h[i] - RING_HB
            dh = -(1 - 2 * s) * field            # energy change of flipping s -> 1 - s
            acc = (dh < 0) | (u[m] < np.exp(-RING_BETA * np.clip(dh, 0, None)))
            spins[ar[acc], i[acc]] = 1 - s[acc]
        out[:, k] = spins
    return out


# ----------------------------------------------------------------------------
# Per-cycle table
# ----------------------------------------------------------------------------
def cycle_table(root, run):
    """One row per logged control cycle of one run."""
    ring = load_ring(root, run)
    n = ring["n"]
    t0 = ring["t"].iloc[0]
    ts = (ring["t"] - t0).dt.total_seconds().to_numpy()
    tod = (ring["t"].dt.hour * 3600 + ring["t"].dt.minute * 60 + ring["t"].dt.second
           + ring["t"].dt.microsecond * 1e-6).to_numpy()

    rows = []
    prevL = prevR = None
    for k in range(n):
        lb = lobes(ring["hext"][k])[:2]
        thL = qL = thR = qR = np.nan
        if len(lb) == 2:
            (a1, v1), (a2, v2) = lb
            (thL, qL), (thR, qR) = ((a1, v1), (a2, v2)) if a1 > a2 else ((a2, v2), (a1, v1))
        elif len(lb) == 1:
            # one lobe only: decide which speaker it is from the previous cycle's lobes,
            # shifted by the turn the robot made in between
            a, v = lb[0]
            turn = ring["bump"][k - 1] if k > 0 else 0.0
            if prevL is None or abs(wrap(a - (prevL - turn))) <= abs(wrap(a - (prevR - turn))):
                thL, qL = a, v
            else:
                thR, qR = a, v
        if np.isfinite(thL) and np.isfinite(thR):
            prevL, prevR = thL, thR
        s = ring["spins"][k]

        def on(a):
            return bool(s[int(np.round((a + 180.0) / 3.0)) % N]) if np.isfinite(a) else False
        rows.append(dict(run=run, k=k, t=ts[k], tod=tod[k], bump=ring["bump"][k],
                         thL=thL, qL=qL, thR=thR, qR=qR,
                         two_lobes=bool(np.isfinite(thL) and np.isfinite(thR)),
                         coverL=on(thL), coverR=on(thR), n_active=int(s.sum())))
    T = pd.DataFrame(rows)

    # commitment.  k_split: the bump stops spanning both lobes for good.  kdec: the cycle,
    # at most three before k_split, whose bump first turns the robot by more than TURN_DEG
    # towards one speaker.  Every input before kdec was heard with the robot still heading
    # between the two speakers.
    both = (T.coverL & T.coverR).to_numpy()
    k_split = int(np.where(both)[0].max()) + 1
    kdec = k_split
    for k in range(max(k_split - 3, 0), k_split + 1):
        if abs(T.bump.iloc[k]) > TURN_DEG:
            kdec = k
            break
    choice = "L" if T.bump.iloc[kdec] > 0 else "R"
    T["kdec"] = kdec
    T["k_split"] = k_split
    T["choice"] = choice
    T["level_R"] = RIGHT_LEVEL[run]
    T["session"] = SESSION[run]
    T["pre"] = T.k < kdec
    T["dropout"] = T.pre & ~T.two_lobes

    T["rho"] = T.qR / T.qL                               # right / left lobe height
    T["ell"] = np.log(T.rho)
    T["o"] = T.qL / np.maximum(T.qL, T.qR) - T.qR / np.maximum(T.qL, T.qR)   # q_L - q_R, taller lobe = 1
    T["alpha"] = T.thL - T.thR                           # lobe separation, deg
    T["m"] = 0.5 * (T.thL + T.thR)                       # bisector of the lobes in the robot frame

    # geometry from Qualisys (clocks of the robot and of the tracker agree to < 0.1 s)
    for c in ["x", "y", "dL", "dR", "ldr", "alphaQ", "still_mm"]:
        T[c] = np.nan
    tr = load_traj(root, run)
    if tr is not None:
        tt = (tr["t"] - t0).dt.total_seconds().to_numpy()
        X = tr[["robot_x", "robot_y"]].to_numpy(float)
        SL = np.nanmedian(tr[["speakerL_x", "speakerL_y"]].to_numpy(float), axis=0)
        SR = np.nanmedian(tr[["speakerR_x", "speakerR_y"]].to_numpy(float), axis=0)
        for k in range(n):
            # the robot stands still while it listens
            msk = (tt >= ts[k] - ACQ_S + 0.3) & (tt <= ts[k] - 0.1) & np.isfinite(X[:, 0])
            if msk.sum() < 100:
                continue
            p = np.median(X[msk], axis=0)
            vL, vR = SL - p, SR - p
            T.loc[k, ["x", "y"]] = p
            T.loc[k, "still_mm"] = float(np.hypot(*X[msk].std(axis=0)))
            T.loc[k, "dL"], T.loc[k, "dR"] = np.hypot(*vL), np.hypot(*vR)
            T.loc[k, "alphaQ"] = float(wrap(np.degrees(np.arctan2(vL[1], vL[0]) - np.arctan2(vR[1], vR[0]))))
        T["ldr"] = np.log(T.dL / T.dR)
    T["tracked"] = np.isfinite(T.ldr)
    return T, ring


def clock_check(root, run, ring_t):
    """Median delay between a logged cycle and the start of the robot's next move in the
    Qualisys record.  The controller logs, turns and drives off at once, so a delay of a
    few hundredths of a second means that the two computers' clocks agree."""
    tr = load_traj(root, run)
    if tr is None:
        return np.nan
    t0 = ring_t.iloc[0]
    tt = (tr["t"] - t0).dt.total_seconds().to_numpy()
    ts = (ring_t - t0).dt.total_seconds().to_numpy()
    X = tr[["robot_x", "robot_y"]].to_numpy(float)
    ok = np.isfinite(X[:, 0])
    tu = np.arange(tt[ok][0], tt[ok][-1], 0.01)
    xu, yu = np.interp(tu, tt[ok], X[ok, 0]), np.interp(tu, tt[ok], X[ok, 1])
    ker = np.ones(15) / 15.0                                   # 0.15 s moving average
    xs, ys = np.convolve(xu, ker, "same"), np.convolve(yu, ker, "same")
    moving = np.hypot(np.gradient(xs, 0.01), np.gradient(ys, 0.01)) > 8.0      # mm/s
    onset = tu[1:][moving[1:] & ~moving[:-1]]
    lags = []
    for t in ts:
        j = np.searchsorted(onset, t - 0.5)
        if j < len(onset) and onset[j] - t < 1.5:
            lags.append(onset[j] - t)
    return float(np.median(lags)) if lags else np.nan


# ----------------------------------------------------------------------------
# Route A: regression on the measured geometry
# ----------------------------------------------------------------------------
PAR_NAMES = ["c_0.7", "c_0.6", "a1", "a2", "a3", "gamma0", "gamma1"]


def design(alpha, ldr, level):
    z = (np.asarray(alpha, float) - ALPHA_REF) / ALPHA_SCALE
    ldr = np.asarray(ldr, float)
    level = np.asarray(level, float)
    return np.column_stack([(level == 0.7) * 1.0, (level == 0.6) * 1.0, z, z ** 2, z ** 3, ldr, ldr * z])


def huber(X, y, c=1.345, iters=100):
    """Huber M-estimate by iteratively reweighted least squares; returns (beta, scale)."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s = 1.0
    for _ in range(iters):
        r = y - X @ b
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12))
        sw = np.sqrt(w)
        b2 = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(b2 - b)) < 1e-12:
            b = b2
            break
        b = b2
    return b, s


def fit_route_a(P):
    """P: pre-commitment, two-lobe, tracked cycles.  Model for ell = ln(q_R/q_L):
        ell = c_level + a1 z + a2 z^2 + a3 z^3 + (gamma0 + gamma1 z) ln(d_L/d_R),  z = (alpha-70)/25."""
    X = design(P.alpha, P.ldr, P.level_R)
    beta, scale = huber(X, P.ell.to_numpy())
    return beta, scale


def predict_ell(beta, alpha, ldr, level):
    return design(alpha, ldr, level) @ beta


def o_of_ell(ell):
    """q_L - q_R with the taller lobe equal to one, as a function of ell = ln(q_R/q_L)."""
    ell = np.asarray(ell, float)
    return np.where(ell <= 0, 1.0 - np.exp(ell), np.exp(-ell) - 1.0)


# ----------------------------------------------------------------------------
# Route B: successive differences
# ----------------------------------------------------------------------------
def successive_diffs(df, col):
    """Differences between consecutive cycles (k, k+1) of the same run, with each
    run's mean difference removed (a constant drift inside the window is not noise)."""
    out = []
    for _, d in df.groupby("run", sort=False):
        d = d.sort_values("k")
        v, k = d[col].to_numpy(), d.k.to_numpy()
        dd = (v[1:] - v[:-1])[(k[1:] - k[:-1]) == 1]
        if len(dd) >= 2:
            out.append(dd - dd.mean())
    return out


def sd_from_diffs(groups):
    """Noise s.d. from drift-corrected successive differences, pooled over runs:
    classical (pooled variance / 2) and robust (MAD)."""
    m = sum(len(g) for g in groups)
    if m - len(groups) < 3:
        return np.nan, np.nan
    ss = sum(float(np.sum(g ** 2)) for g in groups)
    classical = math.sqrt(ss / (m - len(groups)) / 2.0)
    allv = np.concatenate(groups)
    robust = 1.4826 * np.median(np.abs(allv - np.median(allv))) / math.sqrt(2.0)
    return classical, robust


def sd_from_second_diffs(df, col):
    """Check on Route B: noise s.d. from second differences, which also remove a
    linear trend:  var(o[n+1] - 2 o[n] + o[n-1]) = 6 sigma^2."""
    dd = []
    for _, d in df.groupby("run", sort=False):
        d = d.sort_values("k")
        v, k = d[col].to_numpy(), d.k.to_numpy()
        for i in range(1, len(d) - 1):
            if k[i + 1] - k[i] == 1 and k[i] - k[i - 1] == 1:
                dd.append(v[i + 1] - 2.0 * v[i] + v[i - 1])
    return math.sqrt(np.mean(np.square(dd)) / 6.0) if len(dd) >= 4 else np.nan


def mad_sd(r):
    r = np.asarray(r, float)
    return 1.4826 * np.median(np.abs(r - np.median(r)))


# ----------------------------------------------------------------------------
# Windows and the SNR table
# ----------------------------------------------------------------------------
WINDOWS = ("start", "pre-commitment", "whole approach")
GROUPS = ("0.7R", "0.7L", "0.6L")          # right level + side chosen
STAT_COLS = ("dqA", "dqB", "sigA", "sigB", "sigA_rob", "sigB_rob",
             "dprimeA", "dprimeB", "DA", "DB", "DA_rob", "DB_rob")


def in_window(k, kdec, window):
    if window == "start":
        return k < START_CYCLES
    if window == "pre-commitment":
        return k >= kdec - COMMIT_CYCLES
    return np.ones(len(k), bool)


def snr_row(W, dt):
    """Signal, noise and SNR of one window by both routes.

    W: valid (two-lobe) pre-commitment cycles.  Columns fit_o / res_o hold the
    Route A fitted value and residual of o where the robot was tracked."""
    A = W[np.isfinite(W.res_o)]
    row = dict(n=len(W), nA=len(A), rho=float(np.exp(W.ell.mean())))
    # Route A: signal = fitted mean, noise = residual
    row["dqA"] = float(A.fit_o.mean()) if len(A) > 5 else np.nan
    row["sigA"] = float(np.sqrt(np.mean(A.res_o ** 2))) if len(A) > 5 else np.nan
    row["sigA_rob"] = float(mad_sd(A.res_o)) if len(A) > 5 else np.nan
    # Route B: signal = window mean, noise = successive differences
    row["dqB"] = float(W.o.mean())
    row["sigB"], row["sigB_rob"] = sd_from_diffs(successive_diffs(W, "o"))
    row["sigB_2nd"] = sd_from_second_diffs(W, "o")          # check: second differences
    row["eta_eqB"] = row["sigB"] / math.sqrt(2.0)           # equal noise on both targets
    for r in "AB":
        for suf in ("", "_rob"):
            s, dq = row[f"sig{r}{suf}"], row[f"dq{r}"]
            dp = dq / s if (s and np.isfinite(s)) else np.nan
            row[f"dprime{r}{suf}"] = dp
            row[f"D{r}{suf}"] = math.sqrt(2.0) * dp
    row["pB"] = Phi(abs(row["dprimeB"])) if np.isfinite(row["dprimeB"]) else np.nan
    row["AcB"] = row["dprimeB"] / math.sqrt(dt)
    row["frac_contradict"] = float(np.mean(np.sign(W.o) != np.sign(row["dqB"])))
    return row


def add_route_a(V, beta):
    """Fitted value and residual of o for tracked cycles."""
    V = V.copy()
    fit = o_of_ell(predict_ell(beta, V.alpha, V.ldr.fillna(0.0), V.level_R))
    V["fit_ell"] = np.where(V.tracked, predict_ell(beta, V.alpha, V.ldr.fillna(0.0), V.level_R), np.nan)
    V["fit_o"] = np.where(V.tracked, fit, np.nan)
    V["res_o"] = V.o - V.fit_o
    V["res_ell"] = V.ell - V.fit_ell
    return V


def snr_table(V, dt):
    rows = []
    kk, kd = V.k_orig.to_numpy(), V.kdec.to_numpy()
    # the five runs at right level 0.7 share the start; they part ways later
    W = V[(V.level_R == 0.7) & in_window(kk, kd, "start")]
    rows.append(dict(group="0.7", window="start", **snr_row(W, dt)))
    for g in GROUPS:
        for w in WINDOWS:
            W = V[(V.grp == g) & in_window(kk, kd, w)]
            if len(W) < 6:
                continue
            rows.append(dict(group=g, window=w, **snr_row(W, dt)))
    return pd.DataFrame(rows)


def approach_curve(V, dt, block=BLOCK_CYCLES, n_blocks=7):
    """Signal, noise and SNR in blocks of `block` cycles counted back from the turn, so
    that the runs are aligned where the mean changes fastest.  Block 0 is the last
    `block` cycles before the turn; the first cycles of a run fall in the last block."""
    V = V.copy()
    V["block"] = np.minimum((V.kdec - 1 - V.k_orig) // block, n_blocks - 1)
    rows = []
    for g in GROUPS:
        for b in range(n_blocks):
            W = V[(V.grp == g) & (V.block == b)]
            if len(W) < 5:
                continue
            rows.append(dict(group=g, block=b, alpha=float(W.alpha.mean()), alpha_lo=float(W.alpha.min()),
                             alpha_hi=float(W.alpha.max()), **snr_row(W, dt)))
    return pd.DataFrame(rows).sort_values(["group", "alpha"]).reset_index(drop=True)


def derived_quantities(beta, start_ldr):
    """Model summaries: distance exponent and the lobe ratio at chosen poses."""
    out = {}
    for a in (50.0, 70.0, 95.0):
        z = (a - ALPHA_REF) / ALPHA_SCALE
        out[f"gamma@{a:.0f}"] = float(beta[5] + beta[6] * z)
    for lev in (0.7, 0.6):
        for a in (50.0, 95.0):
            out[f"rho_bisector_{lev}@{a:.0f}"] = float(np.exp(predict_ell(beta, [a], [0.0], [lev])[0]))
        out[f"rho_start_{lev}@50"] = float(np.exp(predict_ell(beta, [50.0], [start_ldr], [lev])[0]))
    out["level_step_ln"] = float(beta[0] - beta[1])
    return out


def resample(V, rng):
    """Two-stage bootstrap sample: runs within group, then consecutive pairs within run."""
    parts = []
    for g in GROUPS:
        runs = sorted(V[V.grp == g].run.unique())
        for j, r in enumerate(rng.choice(runs, size=len(runs), replace=True)):
            d = V[V.run == r].sort_values("k")
            kk = d.k.to_numpy()
            ok = np.where(kk[1:] - kk[:-1] == 1)[0]          # first index of a consecutive pair
            i = rng.choice(ok, size=max(len(d) // 2, 1), replace=True)
            pair = pd.concat([d.iloc[i], d.iloc[i + 1]])
            pair = pair.assign(k=np.r_[3 * np.arange(len(i)), 3 * np.arange(len(i)) + 1],
                               run=f"{r}#{g}{j}")
            parts.append(pair)
    return pd.concat(parts, ignore_index=True)


def bootstrap(V, dt, start_ldr, n_boot, seed=2026):
    rng = np.random.default_rng(seed)
    tabs, betas, ders = [], [], []
    for _ in range(n_boot):
        B = resample(V, rng)
        Bt = B[B.tracked]
        beta, _ = fit_route_a(Bt)
        B = add_route_a(B, beta)
        tabs.append(snr_table(B, dt).set_index(["group", "window"])[list(STAT_COLS)])
        betas.append(beta)
        ders.append(derived_quantities(beta, start_ldr))
    return tabs, np.array(betas), pd.DataFrame(ders)


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if len(a) else (np.nan, np.nan)


# ----------------------------------------------------------------------------
# Diagnostics of the noise
# ----------------------------------------------------------------------------
def acf_within_runs(V, col, max_lag=5):
    """Autocorrelation of a residual series, pooled over runs (pairs of the same run only)."""
    out = []
    v0 = V[col].to_numpy()
    var = np.nanmean((v0 - np.nanmean(v0)) ** 2)
    for lag in range(1, max_lag + 1):
        num, cnt = 0.0, 0
        for _, d in V.groupby("run", sort=False):
            d = d.sort_values("k")
            v, k = d[col].to_numpy(), d.k.to_numpy()
            for i in range(len(d) - lag):
                if k[i + lag] - k[i] == lag and np.isfinite(v[i]) and np.isfinite(v[i + lag]):
                    num += (v[i] - np.nanmean(v0)) * (v[i + lag] - np.nanmean(v0))
                    cnt += 1
        out.append(dict(lag=lag, r=num / cnt / var, n=cnt, se=1.0 / math.sqrt(cnt)))
    return pd.DataFrame(out)


def phase_locking(V, col, n_bins=8, n_perm=5000, seed=1):
    """Share of the residual variance explained by the phase of the cycle in the 2-s
    chirp period, per sound session, with a permutation p-value."""
    rng = np.random.default_rng(seed)
    out = []
    for s, d in V.groupby("session"):
        r = d[col].to_numpy()
        ok = np.isfinite(r)
        r, ph = r[ok], (d.tod.to_numpy()[ok] % 2.0)
        b = np.minimum((ph / 2.0 * n_bins).astype(int), n_bins - 1)

        def eta2(rr):
            tot = np.sum((rr - rr.mean()) ** 2)
            between = sum(np.sum(b == j) * (rr[b == j].mean() - rr.mean()) ** 2 for j in range(n_bins) if np.any(b == j))
            return between / tot
        e = eta2(r)
        perm = np.array([eta2(rng.permutation(r)) for _ in range(n_perm)])
        out.append(dict(session=int(s), n=len(r), eta2=float(e), p=float((np.sum(perm >= e) + 1) / (n_perm + 1)),
                        eta2_null=float(perm.mean())))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def analyse(root, n_boot=2000, n_seeds=200):
    tabs, rings = [], {}
    for run in RUNS:
        T, ring = cycle_table(root, run)
        tabs.append(T)
        rings[run] = ring
    C = pd.concat(tabs, ignore_index=True)
    C["grp"] = C.level_R.map("{:.1f}".format) + C.choice
    C["k_orig"] = C.k
    dt = float(np.median(np.concatenate([np.diff(C[C.run == r].t) for r in RUNS])))

    # ---- inventory and integrity checks --------------------------------------------------
    inv = []
    for run in RUNS:
        d = C[C.run == run]
        wrong, dbump = replay_check(rings[run])
        pre = d[d.pre]
        inv.append(dict(run=f"A1_{run}", level_R=RIGHT_LEVEL[run], cycles=len(d), kdec=int(d.kdec.iloc[0]),
                        k_split=int(d.k_split.iloc[0]), turn_deg=float(d.bump.iloc[int(d.kdec.iloc[0])]),
                        choice_from_log=d.choice.iloc[0], choice_reported=REPORTED_CHOICE[run],
                        pre_cycles=len(pre), dropouts=int(pre.dropout.sum()),
                        tracked_pre=int(pre.tracked.sum()), tracked_all=int(d.tracked.sum()),
                        replay_wrong_spins=wrong, replay_max_bump_err=dbump,
                        motion_lag_s=clock_check(root, run, rings[run]["t"]),
                        alpha_at_commit=float(d[d.k == d.kdec.iloc[0] - 1].alpha.iloc[0])))
    inv = pd.DataFrame(inv)

    # ---- valid pre-commitment samples ----------------------------------------------------
    V = C[C.pre & C.two_lobes].copy()
    beta, hub_scale = fit_route_a(V[V.tracked])
    V = add_route_a(V, beta)
    start_ldr = float(V[(V.k < 3) & V.tracked].ldr.mean())
    table = snr_table(V, dt)
    der = derived_quantities(beta, start_ldr)
    st = V[V.k < START_CYCLES].groupby("level_R").ell.mean()
    slope = (st[0.7] - st[0.6]) / 0.1                       # d ln(rho) / d level, two points only
    level_cal = dict(ln_rho_start_07=float(st[0.7]), ln_rho_start_06=float(st[0.6]), slope_per_level=float(slope),
                     level_equal_lobes_extrapolated=float(0.7 - st[0.7] / slope))

    # ---- bootstrap -----------------------------------------------------------------------
    tabs_b, betas_b, ders_b = bootstrap(V, dt, start_ldr, n_boot)
    for c in STAT_COLS:
        lo, hi = [], []
        for _, r in table.iterrows():
            vals = [t.loc[(r.group, r.window), c] for t in tabs_b if (r.group, r.window) in t.index]
            a, b = ci(vals)
            lo.append(a)
            hi.append(b)
        table[c + "_lo"], table[c + "_hi"] = lo, hi
    par = pd.DataFrame(dict(name=PAR_NAMES, value=beta,
                            lo=[ci(betas_b[:, j])[0] for j in range(len(beta))],
                            hi=[ci(betas_b[:, j])[1] for j in range(len(beta))]))
    der_ci = {k: dict(value=v, lo=ci(ders_b[k])[0], hi=ci(ders_b[k])[1]) for k, v in der.items()}

    # ---- noise diagnostics (log-ratio residuals of Route A) -------------------------------
    R = V[np.isfinite(V.res_ell)]
    res = R.res_ell.to_numpy()
    rob = mad_sd(res)
    core = res[np.abs(res - np.median(res)) <= 3 * rob]
    zc = (res - res.mean()) / res.std()
    diag = dict(
        n=int(len(res)), sd=float(res.std(ddof=len(beta))), robust_sd=float(rob), huber_scale=float(hub_scale),
        skew=float(np.mean(zc ** 3)), excess_kurtosis=float(np.mean(zc ** 4) - 3.0),
        outliers_3sigma=int(len(res) - len(core)), sd_core=float(core.std(ddof=1)),
        dropouts=int(C[C.pre].dropout.sum()), pre_cycles=int(C.pre.sum()))
    try:                                   # Shapiro-Wilk needs scipy; everything else does not
        from scipy import stats
        diag["shapiro_p"] = float(stats.shapiro(res).pvalue)
        diag["shapiro_p_core"] = float(stats.shapiro(core).pvalue)
    except ImportError:
        pass
    # the robot's own heading steps: change of ln(q_R/q_L) against change of the lobe bisector
    for name, sel in [("start", V.k < START_CYCLES), ("whole", V.k >= 0)]:
        dl, dm = [], []
        for _, d in V[sel].groupby("run"):
            d = d.sort_values("k")
            ok = np.diff(d.k.to_numpy()) == 1
            dl += list(np.diff(d.ell.to_numpy())[ok])
            dm += list(np.diff(d.m.to_numpy())[ok])
        r = float(np.corrcoef(dl, dm)[0, 1])
        diag[f"heading_coupling_{name}"] = dict(n=len(dl), corr=r, slope_per_deg=float(np.polyfit(dm, dl, 1)[0]),
                                                variance_share=r * r, sd_heading_step_deg=float(np.std(dm)))
    acf = acf_within_runs(R, "res_ell")
    # detrended series for all runs (running median of 9), used for the phase test
    V["detr_ell"] = np.nan
    for r, d in V.groupby("run"):
        med = d.sort_values("k").ell.rolling(9, center=True, min_periods=4).median()
        V.loc[med.index, "detr_ell"] = d.ell - med
    phase = phase_locking(V, "detr_ell")
    # lobe separation against the angle the speakers subtend at the tracked position
    tr = C[C.tracked & C.two_lobes]
    diag["alpha_minus_alphaQ_pre_mean"] = float((tr[tr.pre].alpha - tr[tr.pre].alphaQ).mean())
    diag["alpha_minus_alphaQ_pre_sd"] = float((tr[tr.pre].alpha - tr[tr.pre].alphaQ).std())
    diag["corr_alpha_alphaQ"] = float(np.corrcoef(tr.alpha, tr.alphaQ)[0, 1])
    diag["still_mm_max"] = float(C.still_mm.max())

    # ---- what decided each run -----------------------------------------------------------
    dec = []
    for run in RUNS:
        d = V[V.run == run]
        last = d[d.k >= d.kdec - 3]
        dec.append(dict(run=f"A1_{run}", level_R=RIGHT_LEVEL[run], choice=d.choice.iloc[0],
                        o_last3=float(last.o.mean()), rho_last3=float(np.exp(last.ell.mean())),
                        stronger_lobe="L" if last.o.mean() > 0 else "R",
                        o_first15=float(d[d.k < START_CYCLES].o.mean()),
                        ldr_start=float(d[d.k < 3].ldr.mean()), ldr_last3=float(last.ldr.mean()),
                        y_last3=float(last.y.mean())))
    dec = pd.DataFrame(dec)
    dec["chose_stronger"] = dec.choice == dec.stronger_lobe

    # ---- does the ring's own randomness matter?  replay with other seeds ------------------
    seeds = []
    th = np.linspace(-np.pi, np.pi, N, endpoint=False)
    if n_seeds:
        for run in RUNS:
            d = C[C.run == run]
            kdec = int(d.kdec.iloc[0])
            ncyc = kdec + 1
            sp = replay_seeds(rings[run]["hext"], ncyc, list(range(1000, 1000 + n_seeds)))
            bump = np.degrees(np.angle(np.einsum("skn,n->sk", sp.astype(float), np.exp(1j * th))))
            # state of each replica at the cycle of the logged turn: its bump has turned
            # towards the left lobe, towards the right lobe, or is still between the two
            b = bump[:, kdec]
            side = np.where(b > TURN_DEG, "L", np.where(b < -TURN_DEG, "R", "-"))
            seeds.append(dict(run=f"A1_{run}", n_seeds=n_seeds, logged_choice=d.choice.iloc[0],
                              same_side=float(np.mean(side == d.choice.iloc[0])),
                              other_side=float(np.mean((side != d.choice.iloc[0]) & (side != "-"))),
                              not_yet=float(np.mean(side == "-")), logged_turn_cycle=kdec))
    seeds = pd.DataFrame(seeds)

    # ---- sensitivity of Route B to the window length ---------------------------------------
    sens = []
    for g in ("0.7R", "0.6L"):
        G = V[V.grp == g]
        for w, lens in (("start", (8, 10, 15, 20, 25)), ("pre-commitment", (5, 8, 12))):
            for n_w in lens:
                W = G[G.k < n_w] if w == "start" else G[G.k >= G.kdec - n_w]
                s, s_rob = sd_from_diffs(successive_diffs(W, "o"))
                sens.append(dict(group=g, window=w, cycles=n_w, n=len(W), dq=float(W.o.mean()), sigB=s,
                                 sigB_rob=s_rob, DB=math.sqrt(2.0) * W.o.mean() / s))
    sens = pd.DataFrame(sens)

    # ---- signal, noise and SNR along the approach ------------------------------------------
    curve = approach_curve(V, dt)

    return dict(cycles=C, valid=V, inventory=inv, table=table, params=par, derived=der_ci, diag=diag, curve=curve,
                level_cal=level_cal, sensitivity=sens,
                acf=acf, phase=phase, decisions=dec, seeds=seeds, beta=beta, dt=dt, start_ldr=start_ldr)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(here), help="repository root (with ringStats/ and trajectories/)")
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    R = analyse(a.root, a.boot, a.seeds)

    cols = ["run", "k", "t", "tod", "session", "level_R", "choice", "grp", "kdec", "k_split", "pre", "two_lobes", "dropout", "bump", "n_active",
            "thL", "qL", "thR", "qR", "rho", "ell", "o", "alpha", "m", "tracked", "x", "y", "dL", "dR", "ldr", "alphaQ"]
    R["cycles"][cols].to_csv(os.path.join(a.out, "cycles.csv"), index=False, float_format="%.6g")
    R["valid"][["run", "k", "grp", "session", "tod", "kdec", "o", "ell", "alpha", "ldr", "fit_o", "res_o", "fit_ell", "res_ell", "detr_ell"]].to_csv(
        os.path.join(a.out, "precommit_samples.csv"), index=False, float_format="%.6g")
    R["inventory"].to_csv(os.path.join(a.out, "runs.csv"), index=False, float_format="%.6g")
    R["table"].to_csv(os.path.join(a.out, "snr_table.csv"), index=False, float_format="%.5g")
    R["params"].to_csv(os.path.join(a.out, "route_a_parameters.csv"), index=False, float_format="%.5g")
    R["decisions"].to_csv(os.path.join(a.out, "decisions.csv"), index=False, float_format="%.5g")
    R["curve"].to_csv(os.path.join(a.out, "snr_along_approach.csv"), index=False, float_format="%.5g")
    R["sensitivity"].to_csv(os.path.join(a.out, "window_sensitivity.csv"), index=False, float_format="%.5g")
    R["acf"].to_csv(os.path.join(a.out, "noise_acf.csv"), index=False, float_format="%.5g")
    R["phase"].to_csv(os.path.join(a.out, "noise_phase_locking.csv"), index=False, float_format="%.5g")
    if len(R["seeds"]):
        R["seeds"].to_csv(os.path.join(a.out, "seed_replay.csv"), index=False, float_format="%.5g")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(cycle_period_s=R["dt"], start_ln_dL_over_dR=R["start_ldr"], derived=R["derived"],
                       level_calibration=R["level_cal"], diagnostics=R["diag"]), f, indent=2)

    print("\nRUNS\n", R["inventory"].to_string(index=False))
    show = ["group", "window", "n", "nA", "rho", "dqA", "dqB", "sigA", "sigB", "sigB_2nd", "sigA_rob", "sigB_rob",
            "eta_eqB", "dprimeA", "dprimeB", "DA", "DB", "DA_rob", "DB_rob", "pB", "AcB", "frac_contradict"]
    print("\nSNR TABLE (o = q_L - q_R; D = sqrt(2) Dq / sigma_o)\n", R["table"][show].round(4).to_string(index=False))
    print("\n95% intervals\n", R["table"][["group", "window", "dqB_lo", "dqB_hi", "sigA_lo", "sigA_hi", "sigB_lo", "sigB_hi",
                                            "DA_lo", "DA_hi", "DB_lo", "DB_hi"]].round(4).to_string(index=False))
    print("\nROUTE A PARAMETERS\n", R["params"].round(4).to_string(index=False))
    print("\nDERIVED\n", json.dumps(R["derived"], indent=1))
    print("\nLEVEL CALIBRATION\n", json.dumps(R["level_cal"], indent=1))
    print("\nDIAGNOSTICS\n", json.dumps(R["diag"], indent=1))
    print("\nACF\n", R["acf"].round(3).to_string(index=False))
    print("\nPHASE LOCKING\n", R["phase"].round(4).to_string(index=False))
    print("\nWINDOW SENSITIVITY (Route B)\n", R["sensitivity"].round(4).to_string(index=False))
    print("\nALONG THE APPROACH\n", R["curve"][["group", "block", "alpha", "n", "nA", "dqA", "dqB", "sigA", "sigB",
                                                    "DA", "DB"]].round(4).to_string(index=False))
    print("\nDECISIONS\n", R["decisions"].round(4).to_string(index=False))
    if len(R["seeds"]):
        print("\nSEED REPLAY\n", R["seeds"].round(3).to_string(index=False))

    T = R["table"].set_index(["group", "window"])
    print("\nHEADLINE: per-sample SNR in the ICRA convention, D = Dq / eta = sqrt(2) Dq / sigma_o")
    for g, w, name in [("0.7", "start", "right 0.7, start (cycles 0-14)"),
                       ("0.6L", "start", "right 0.6, start (cycles 0-14)"),
                       ("0.7R", "pre-commitment", "right 0.7, last 8 cycles, runs that turned right"),
                       ("0.6L", "pre-commitment", "right 0.6, last 8 cycles")]:
        r = T.loc[(g, w)]
        print(f"  {name:50s} Route A: D = {r.DA:6.1f} [{r.DA_lo:6.1f}, {r.DA_hi:6.1f}]   "
              f"Route B: D = {r.DB:6.1f} [{r.DB_lo:6.1f}, {r.DB_hi:6.1f}]")


if __name__ == "__main__":
    main()
