"""Entrenamiento de las redes: covariables de calendario, AdamW, Huber, scheduler,
early stopping con validación cronológica y ensamble de varias semillas."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 
from utils import progress
from utils.features import COV_COLS, calendar_features
from .pytorch_arch import build
 
L, SEEDS, EPOCHS, PATIENCE = 56, 3, 80, 10
 
 
def _train(kind, xx, yy, seed, epochs=EPOCHS, on_epoch=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    nv = max(int(len(xx) * 0.1), 1)  # último 10 % como validación (orden temporal)
    xt, yt, xv, yv = xx[:-nv], yy[:-nv], torch.tensor(xx[-nv:]), torch.tensor(yy[-nv:])
    dl = DataLoader(TensorDataset(torch.tensor(xt), torch.tensor(yt)), batch_size=64, shuffle=True)
    net = build(kind, xx.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lf, best, bad, state = nn.HuberLoss(), 1e9, 0, None
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            lf(net(xb), yb).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lf(net(xv), yv).item()
        sch.step(v)
        if on_epoch:
            on_epoch((ep + 1) / epochs)
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                if on_epoch:
                    on_epoch(1.0)  # paró antes: se da por completada esta red
                break
    net.load_state_dict(state)
    return net.eval()
 
 
def fit_predict(kind, y, idx, H, country=None):
    n = len(y)
    fut = pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=H)
    C = calendar_features(idx.append(fut), country)[COV_COLS].values.astype(np.float32)
    log = np.log1p(np.maximum(y, 0))
    mu, sd = log.mean(), log.std() + 1e-9
    z = ((log - mu) / sd).astype(np.float32)
    # cada paso lleva el valor del día y las covariables del día SIGUIENTE (ya conocidas)
    seq = lambda zv, t: np.concatenate([np.asarray(zv[t - L:t], np.float32)[:, None], C[t - L + 1:t + 1]], 1)
    xx = np.stack([seq(z, t) for t in range(L, n)])
    yy = z[L:, None]
    preds = []
    seeds, epochs = (1, 40) if progress.quick() else (SEEDS, EPOCHS)  # modo rápido: 1 red, menos épocas
    for s in range(seeds):  # ensamble: se promedian los pronósticos de varias redes
        def avance(f, s=s):
            progress.report((s + f) / seeds, f"Entrenando red {s + 1} de {seeds}…")
        net, h = _train(kind, xx, yy, s, epochs, avance), list(z)
        with torch.no_grad():
            for k in range(H):
                h.append(net(torch.tensor(seq(h, n + k)[None])).item())
        preds.append(np.expm1(np.array(h[n:]) * sd + mu))
    return np.maximum(0, np.mean(preds, 0))
 
 
def gru(y, idx, H, country=None):
    return fit_predict("gru", y, idx, H, country)
 
 
def lstm(y, idx, H, country=None):
    return fit_predict("lstm", y, idx, H, country)
 
 
def cnn_gru(y, idx, H, country=None):
    return fit_predict("cnn_gru", y, idx, H, country)
