import os
import numpy as np
import scipy.io as sio
from scipy import signal as sp_signal
from scipy.signal import hilbert
import pywt

# Optional deps
try:
    import pycwt as cwt
    HAS_PYCWT = True
except Exception:
    HAS_PYCWT = False

try:
    from PyEMD import EMD
    HAS_PYEMD = True
except Exception:
    HAS_PYEMD = False

from cp_utils import detrend_and_norm, segment_indices, plot_image_and_save, ensure_dirs, extract_pressure
from cp_config import WIN_SEC, OVERLAP

# ----- Core transforms into normalized arrays -----

def stft_image_arr(x, fs):
    f, t, Zxx = sp_signal.stft(x, fs=fs, nperseg=1024, noverlap=512, window="hann")
    S = np.abs(Zxx)
    S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    return S

def cwt_image_arr(x, fs):
    scales = np.arange(1, 256)
    coefs, freqs = pywt.cwt(x, scales, 'morl', sampling_period=1.0/fs)
    A = np.abs(coefs)
    A = (A - A.min()) / (A.max() - A.min() + 1e-9)
    return A

def wca_image_arr(x, y, fs):
    if not HAS_PYCWT:
        raise RuntimeError("pycwt not installed. Run: pip install pycwt")
    dt = 1.0/fs
    s0 = 2*dt
    dj = 1/12
    J = 8/dj
    WCT, coi, freq, sig = cwt.wcoherence(x, y, dt=dt, s0=s0, dj=dj, J=J)
    A = (WCT - WCT.min()) / (WCT.max() - WCT.min() + 1e-9)
    return A

def hhs_image_arr(x, fs, max_imfs=6, fmax=5000, n_bins_f=256, n_bins_t=256):
    if not HAS_PYEMD:
        raise RuntimeError("PyEMD not installed. Run: pip install EMD-signal")
    imfs = EMD().emd(x)
    imfs = imfs[:max_imfs] if imfs.ndim>1 else np.array([imfs])
    T = len(x); t = np.arange(T)/fs
    H = np.zeros((n_bins_f, n_bins_t), dtype=float)
    f_grid = np.linspace(0, fmax, n_bins_f)
    t_grid = np.linspace(0, t[-1], n_bins_t)

    for IMF in imfs:
        a = np.abs(hilbert(IMF))
        ph = np.unwrap(np.angle(hilbert(IMF)))
        inst_f = np.diff(ph)/(2*np.pi) * fs
        inst_f = np.clip(inst_f, 0, fmax)
        tt = t[1:]
        ti = np.clip(np.searchsorted(t_grid, tt) - 1, 0, n_bins_t-1)
        fi = np.clip(np.searchsorted(f_grid, inst_f) - 1, 0, n_bins_f-1)
        for k in range(len(ti)):
            H[fi[k], ti[k]] += a[k+1]
    H = (H - H.min())/(H.max() - H.min() + 1e-9)
    return H

# ----- Dataset image saving -----

REF_CACHE = {}  # pressure -> (ref_window, fs)

def fetch_ref_window(press, mats):
    if press in REF_CACHE:
        return REF_CACHE[press]
    for m in mats:
        if m["label"] == "Normal" and extract_pressure(m["folder"]) == press:
            d = sio.loadmat(m["mat_path"])
            sig = d["signal"]; fs = float(d["fs"].squeeze())
            w0 = detrend_and_norm(sig[0])
            s,e = segment_indices(len(w0), fs, WIN_SEC, OVERLAP)[0]
            REF_CACHE[press] = (w0[s:e], fs)
            return REF_CACHE[press]
    return None

def save_timefreq_images(sig, fs, label, press, file_stem, out_root, all_mats=None):
    rows = []
    C, N = sig.shape
    ref = fetch_ref_window(press, all_mats) if all_mats is not None else None

    for ch in range(C):
        x = detrend_and_norm(sig[ch])
        for wi, (s,e) in enumerate(segment_indices(N, fs, WIN_SEC, OVERLAP)):
            w = x[s:e]
            # STFT
            S = stft_image_arr(w, fs)
            d1 = os.path.join(out_root, "STFT", label); ensure_dirs(d1)
            p1 = os.path.join(d1, f"{file_stem}__{press}__ch{ch+1}_w{wi:04d}_stft.png")
            plot_image_and_save(S, p1)
            rows.append({"img": p1, "label": label, "transform": "STFT", "pressure": press})

            # CWT
            A = cwt_image_arr(w, fs)
            d2 = os.path.join(out_root, "CWT", label); ensure_dirs(d2)
            p2 = os.path.join(d2, f"{file_stem}__{press}__ch{ch+1}_w{wi:04d}_cwt.png")
            plot_image_and_save(A, p2)
            rows.append({"img": p2, "label": label, "transform": "CWT", "pressure": press})

            # WCA (optional)
            if ref is not None and HAS_PYCWT:
                wR, fsR = ref
                L = min(len(w), len(wR))
                W = wca_image_arr(w[:L], wR[:L], fs)
                d3 = os.path.join(out_root, "WCA", label); ensure_dirs(d3)
                p3 = os.path.join(d3, f"{file_stem}__{press}__ch{ch+1}_w{wi:04d}_wca.png")
                plot_image_and_save(W, p3)
                rows.append({"img": p3, "label": label, "transform": "WCA", "pressure": press})

            # EMD/HHS
            try:
                H = hhs_image_arr(w, fs, fmax=5000)
                d4 = os.path.join(out_root, "EMD", label); ensure_dirs(d4)
                p4 = os.path.join(d4, f"{file_stem}__{press}__ch{ch+1}_w{wi:04d}_emd.png")
                plot_image_and_save(H, p4)
                rows.append({"img": p4, "label": label, "transform": "EMD", "pressure": press})
            except Exception:
                pass

    return rows