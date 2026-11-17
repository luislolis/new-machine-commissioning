#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 9 -- reextracao das bases externas com o checkpoint da rodada 9.

Este .py e a FONTE. O que se abre no Colab e rerun_extract9.ipynb, gerado por
`python make_notebook6.py rerun_extract9`. E o rerun_extract7.py com tres
trocas e nada mais, para que so o checkpoint mude:

    CKPT_NOME  embedding_model_r7_seed42.pth  ->  embedding_model_r9_seed42.pth
    ROOT       /content/rerun7                ->  /content/rerun9
    zips       rerun7_*.zip                   ->  rerun9_*.zip

A arquitetura e a mesma da rodada 7 (max pooling, campo receptivo de 78
amostras), entao a celula A2 carrega o checkpoint novo sem mudar nada.

ORDEM DOS NOTEBOOKS
-------------------
    1. train_rodada9.ipynb    treina, embedda o MAFAULDA, deixa o checkpoint
                              embedding_model_r9_seed42.pth na raiz do Drive
    2. rerun_extract9.ipynb   este: IMS, PRONOSTIA e JNU
    3. analise local          descompactar os zips em code/rerun9/ e rodar
                              code/analise_final_v6.py externas

O QUE EXTRAI
------------
O mesmo da rodada 7: JNU (rep_win e rep_rec), PRONOSTIA (rep_snap e rep_win),
IMS (rep_win, um detector por rolamento).

TEMPO E TAMANHO
---------------
GPU T4, ~1 h. Zips copiados para o Drive:
    rerun9_A.zip              JNU, PRONOSTIA, manifesto            ~90 MB
    rerun9_ims_rep_win.zip    IMS, repeticao + z-score por janela  ~340 MB
"""

# %% ===================================================================== A1
# Setup. Nada aqui treina.

import os, re, json, glob, time, hashlib, shutil, copy, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE
print("Device:", DEVICE)

WINDOW_SIZE = 1024
STRIDE = 512
LAMBDA = 1e-3
QUANT = 0.99
ENROL_FRAC_LEGADO = 0.10             # so para a fidelidade das bases externas

EMB_DTYPE = np.float16
CKPT_SHA_RODADA5 = "0c4e002c9f41d930"   # so para registro
CKPT_NOME = "embedding_model_r9_seed42.pth"
POOL_K, POOL_S, POOL_P = 4, 2, 1
IMS_VARIANTES = ("rep_win",)            # ("rep_snap", "rep_win") para reconferir

ROOT = "/content/rerun9"
SUB = {k: os.path.join(ROOT, k) for k in
       ("A",) + tuple("ims_" + v for v in IMS_VARIANTES)}
for d in SUB.values():
    os.makedirs(d, exist_ok=True)
FIDELIDADE = {}
T_START = time.time()


def tempo():
    return "%.1f min" % ((time.time() - T_START) / 60)


# %% ===================================================================== A2
# Rede e checkpoint. Identico as rodadas anteriores.

class EmbeddingNet1D(nn.Module):
    def __init__(self, in_ch=6, embed_dim=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(POOL_K, stride=POOL_S, padding=POOL_P),      # 1024 -> 512
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(POOL_K, stride=POOL_S, padding=POOL_P),      # 512 -> 256
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(POOL_K, stride=POOL_S, padding=POOL_P),      # 256 -> 128
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, embed_dim)

    def forward(self, x, return_norm=False):
        z = self.feat(x.float()).squeeze(-1)
        z = self.head(z)
        zn = F.normalize(z, dim=1)          # eq. (4): um escalar divide as 128 coordenadas
        return (zn, z.norm(dim=1)) if return_norm else zn


try:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception as e:
    print("Drive nao montado (%s). Seguindo so com /content." % e)


def procurar(nome, raizes=("/content", "/content/drive/MyDrive"), max_dirs=40000):
    """Primeiro caminho encontrado para `nome`, podando pastas de sistema."""
    IGNORAR = {".git", ".ipynb_checkpoints", "__pycache__", "sample_data",
               ".shortcut-targets-by-id", ".Trash", ".tmp.driveupload"}
    vistos = 0
    for raiz in raizes:
        if not os.path.isdir(raiz):
            continue
        for base, dirs, arqs in os.walk(raiz):
            dirs[:] = [d for d in dirs if d not in IGNORAR and not d.startswith(".")]
            vistos += 1
            if vistos > max_dirs:
                print("busca interrompida em %d pastas" % max_dirs)
                return None
            if nome in arqs:
                return os.path.join(base, nome)
    return None


CKPT = procurar(CKPT_NOME) or procurar("embedding_model.pth")
if CKPT is None:
    print("Checkpoint nao encontrado. Envie %s:" % CKPT_NOME)
    from google.colab import files as colab_files
    enviado = colab_files.upload()
    CKPT = "/content/" + next(iter(enviado))

CANON_DIR = os.path.dirname(CKPT)
print("checkpoint:", CKPT)

model = EmbeddingNet1D(in_ch=6, embed_dim=128).to(DEVICE)
try:
    state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
except Exception:
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
if hasattr(state, "state_dict"):
    state = state.state_dict()
model.load_state_dict(state, strict=True)
model.eval()

h = hashlib.sha256()
for k in sorted(state.keys()):
    h.update(k.encode())
    h.update(state[k].detach().cpu().numpy().tobytes())
CKPT_SHA = h.hexdigest()[:16]
print("checkpoint sha256[:16] =", CKPT_SHA)
if CKPT_SHA == CKPT_SHA_RODADA5:
    print("<<< ATENCAO: este e o checkpoint da rodada 5. Voce queria o da rodada 6? >>>")
else:
    print("checkpoint novo (a rodada 5 era %s) -- e o esperado." % CKPT_SHA_RODADA5)

# [RODADA 7] confere a geometria antes de seguir: o state_dict tem chaves
# diferentes das rodadas anteriores, entao um checkpoint antigo falha acima,
# no load_state_dict; aqui conferimos os comprimentos e o campo receptivo.
with torch.no_grad():
    _t = torch.zeros(1, 6, WINDOW_SIZE)
    _comp = [_t.shape[-1]]
    for _c in model.to("cpu").feat:
        _t = _c(_t)
        if isinstance(_c, nn.MaxPool1d):
            _comp.append(_t.shape[-1])
    model.to(DEVICE)
    print("comprimento da serie:", " -> ".join(map(str, _comp)))
    assert _comp == [1024, 512, 256, 128], "geometria do pooling: %s" % _comp
_rf, _salto = 1, 1
for _k, _st in ((9, 1), (POOL_K, POOL_S), (9, 1), (POOL_K, POOL_S), (9, 1), (POOL_K, POOL_S)):
    _rf += (_k - 1) * _salto
    _salto *= _st
print("campo receptivo: %d amostras | IMS %.2f ms | PRONOSTIA %.2f ms | JNU %.2f ms"
      % (_rf, 1000.0 * _rf / 20000, 1000.0 * _rf / 25600, 1000.0 * _rf / 50000))

# copia em CPU, so para a fidelidade do MAFAULDA (o artigo avaliou em CPU)
model_cpu = copy.deepcopy(model).to("cpu").eval()

if DEVICE.type == "cpu":
    print("\nAVISO: runtime em CPU. Runtime > Alterar tipo de runtime > GPU T4.")


# %% ===================================================================== A3
# Funcoes. embed/windowize/mahalanobis/legacy identicas as rodadas anteriores.
# NOVAS: znorm_win, fill_cyclic, fill_zeros.

@torch.no_grad()
def embed_windows_array(Xw, batch=256, net=None, dev=None, with_norm=False):
    net = model if net is None else net
    dev = DEVICE if dev is None else dev
    net.eval()
    embs, nrm = [], []
    for i in range(0, len(Xw), batch):
        xb = torch.tensor(Xw[i:i + batch]).to(dev)
        if with_norm:
            e, n = net(xb, return_norm=True)
            embs.append(e.cpu().numpy()); nrm.append(n.cpu().numpy())
        else:
            embs.append(net(xb).cpu().numpy())
    E = np.vstack(embs) if embs else None
    if not with_norm:
        return E
    return E, (np.concatenate(nrm) if nrm else np.empty(0, np.float32))


def windowize(x, window_size=WINDOW_SIZE, stride=STRIDE):
    """x: (C, T) -> (N, C, window_size)"""
    C, T = x.shape
    wins = [x[:, s:s + window_size] for s in range(0, T - window_size + 1, stride)]
    return np.stack(wins, axis=0) if wins else np.empty((0, C, window_size), np.float32)


def mahalanobis_scores(E, mu, invC):
    D = E - mu
    return np.sqrt(np.sum((D @ invC) * D, axis=1))


def legacy_model(E_fit, lam=LAMBDA, q=QUANT):
    """Enrolment exatamente como no artigo. So para o teste de fidelidade."""
    mu = E_fit.mean(axis=0)
    invC = np.linalg.inv(np.cov(E_fit, rowvar=False) + np.eye(E_fit.shape[1]) * lam)
    thr = float(np.quantile(mahalanobis_scores(E_fit, mu, invC), q))
    return mu, invC, thr


def pct_series(embs, mu, invC, thr):
    return np.array([(mahalanobis_scores(E, mu, invC) > thr).mean() * 100
                     for E in embs])


def compare_to_canon(label, pct_new, csv_name, col="pct_anom"):
    path = os.path.join(CANON_DIR, csv_name)
    if not os.path.exists(path):
        print("   fidelidade %-26s sem %s" % (label, csv_name))
        return None
    canon = pd.read_csv(path)[col].values
    n = min(len(canon), len(pct_new))
    dif = float(np.abs(np.asarray(pct_new)[:n] - canon[:n]).max())
    ok = dif < 1e-6 and len(canon) == len(pct_new)
    print("   fidelidade %-26s %d x %d snaps  max|dif| = %.6f  %s"
          % (label, len(pct_new), len(canon), dif, "IDENTICO" if ok else "<<< DIVERGE"))
    FIDELIDADE[label] = dict(max_dif=dif, n_new=int(len(pct_new)),
                             n_canon=int(len(canon)), identico=bool(ok))
    return dif


def save_emb(outdir, name, embs, z_norm=None, **meta):
    """Uma matriz (janelas x 128) + offsets por snapshot (+ ||z~|| por janela)."""
    offsets = np.cumsum([0] + [len(E) for E in embs]).astype(np.int64)
    E = np.concatenate(embs).astype(EMB_DTYPE)
    path = os.path.join(outdir, name + "_emb.npz")
    extra = {} if z_norm is None else dict(z_norm=np.concatenate(z_norm).astype(EMB_DTYPE))
    np.savez_compressed(path, E=E, offsets=offsets, ckpt=CKPT_SHA,
                        window=WINDOW_SIZE, stride=STRIDE, **extra, **meta)
    print("   salvo %-44s %8d janelas  %6.1f MB"
          % (os.path.relpath(path, ROOT), len(E), os.path.getsize(path) / 1e6))


# [NOVO] normalizacao por janela, IDENTICA a do treino da CNN no MAFAULDA
# (windowize_multichannel_from_csv do notebook de treino).
def znorm_win(Xw):
    """(N, C, L) -> z-score por janela e canal."""
    mean = Xw.mean(axis=2, keepdims=True)
    std = Xw.std(axis=2, keepdims=True) + 1e-8
    return (Xw - mean) / std


# [NOVO] a regra unica de canais
def fill_cyclic(Xw):
    """(N, k, L) -> (N, 6, L); a posicao j recebe o canal j mod k."""
    k = Xw.shape[1]
    return np.ascontiguousarray(Xw[:, [j % k for j in range(6)], :])


# o esquema antigo, so para a ablacao e para a fidelidade
def fill_zeros(Xw):
    """(N, k, L) -> (N, 6, L); os k canais nas primeiras posicoes, o resto zero."""
    out = np.zeros((Xw.shape[0], 6, Xw.shape[2]), dtype=np.float32)
    out[:, :Xw.shape[1], :] = Xw
    return out


def znorm_snap(data):
    """(T, k) -> z-score por canal no snapshot inteiro, como nas rodadas 2-4."""
    return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)


# %% ===================================================================== J
# JNU. Leitores verbatim da rodada 2. rep_win = o esquema de hoje (fidelidade);
# rep_rec = normalizacao pela gravacao inteira.

!git clone -q https://github.com/ClarkGableWang/JNU-Bearing-Dataset.git


def read_jnu_csv_1ch(csv_path):
    df = pd.read_csv(csv_path, header=None, sep=",", engine="python")
    if df.shape[1] == 1:
        df = df[0].astype(str).str.split(",", expand=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.iloc[:, -1].astype(np.float32).values


def windowize_1ch(x, window_size=WINDOW_SIZE, stride=STRIDE):
    wins = []
    for i in range(0, len(x) - window_size + 1, stride):
        w = x[i:i + window_size]
        wins.append(((w - w.mean()) / (w.std() + 1e-8))[None, :])
    return np.stack(wins, axis=0).astype(np.float32) if wins else None


def jnu_rep_win(x):
    Xw = windowize_1ch(x)
    return None if Xw is None else np.repeat(Xw, 6, axis=1)


def jnu_rep_rec(x):
    xn = ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)
    return fill_cyclic(windowize(xn[None, :]))


LAB = {"n": "normal", "ib": "inner", "ob": "outer", "tb": "ball"}
meta_jnu = []
for p in sorted(glob.glob("JNU-Bearing-Dataset/**/*.csv", recursive=True)):
    b = os.path.basename(p)
    m = re.match(r"([a-z]+)(600|800|1000)", b.lower())
    if m and m.group(1) in LAB:
        meta_jnu.append(dict(path=p, label=LAB[m.group(1)], rpm=int(m.group(2))))
meta_jnu = pd.DataFrame(meta_jnu)
print(meta_jnu.groupby(["rpm", "label"]).size())

sig = [read_jnu_csv_1ch(p) for p in meta_jnu["path"]]
_j = {"rep_win": [embed_windows_array(jnu_rep_win(x), with_norm=True) for x in sig],
      "rep_rec": [embed_windows_array(jnu_rep_rec(x), with_norm=True) for x in sig]}
jnu = {v: [e for e, _ in pares] for v, pares in _j.items()}
jnu_nrm = {v: [n for _, n in pares] for v, pares in _j.items()}
del _j

pct_jnu = {}
for rpm in sorted(meta_jnu["rpm"].unique()):
    idx = meta_jnu.index[meta_jnu["rpm"] == rpm]
    norm = [i for i in idx if meta_jnu.loc[i, "label"] == "normal"]
    mu, invC, thr = legacy_model(np.vstack([jnu["rep_win"][i] for i in norm]))
    for i in idx:
        pct_jnu[os.path.basename(meta_jnu.loc[i, "path"])] = float(
            (mahalanobis_scores(jnu["rep_win"][i], mu, invC) > thr).mean() * 100)
cpath = os.path.join(CANON_DIR, "jnu_detailed_results.csv")
if os.path.exists(cpath):
    c_ = pd.read_csv(cpath)
    novo = c_["file"].map(pct_jnu)
    dif = float(np.abs(novo.values - c_["%anom"].values).max())
    ok = dif < 1e-6 and novo.notna().all()
    print("   fidelidade %-26s max|dif| = %.6f  %s"
          % ("jnu_rep_win", dif, "IDENTICO" if ok else "<<< DIVERGE"))
    FIDELIDADE["jnu_rep_win"] = dict(max_dif=dif, identico=bool(ok))

os.makedirs(os.path.join(SUB["A"], "jnu"), exist_ok=True)

# [RODADA 6] features de amplitude do JNU, que faltavam. Mesmas oito bandas de
# oitava do IMS e do PRONOSTIA: de fs/256 a fs/2. O JNU amostra a 50 kHz.
JNU_FS = 50000.0
JNU_BANDS = np.array([0.0] + [JNU_FS / 2.0 / (2 ** k) for k in range(7, -1, -1)])
FEAT_NAMES = ["mean", "std", "rms", "peak", "p2p", "kurtosis", "skewness", "crest"] +              ["band_rms_%d" % i for i in range(8)]


def feats_1ch(x, fs, edges):
    x = np.asarray(x, np.float64)
    m, sd = x.mean(), x.std()
    rms = np.sqrt(np.mean(x ** 2))
    pk = np.abs(x).max()
    z = (x - m) / (sd + 1e-12)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    P = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    bandas = [np.sqrt(P[(freqs >= edges[i]) & (freqs < edges[i + 1])].sum() * 2 / len(x))
              for i in range(8)]
    return [m, sd, rms, pk, x.max() - x.min(), (z ** 4).mean(), (z ** 3).mean(),
            pk / (rms + 1e-12)] + bandas


np.savez_compressed(
    os.path.join(SUB["A"], "jnu", "jnu_feats.npz"),
    feats=np.array([feats_1ch(x, JNU_FS, JNU_BANDS) for x in sig])[:, None, :],
    feat_names=np.array(FEAT_NAMES),
    files=np.array([os.path.basename(p_) for p_ in meta_jnu["path"]]),
    labels=meta_jnu["label"].values.astype(str), rpm=meta_jnu["rpm"].values,
    band_edges_hz=JNU_BANDS, fs=JNU_FS)
print("   features do JNU salvas: %d gravacoes x 1 canal x %d"
      % (len(sig), len(FEAT_NAMES)))

os.makedirs(os.path.join(SUB["A"], "jnu"), exist_ok=True)
for v, embs in jnu.items():
    save_emb(os.path.join(SUB["A"], "jnu"), "jnu_%s" % v, embs, z_norm=jnu_nrm[v],
             files=np.array([os.path.basename(p) for p in meta_jnu["path"]]),
             labels=meta_jnu["label"].values.astype(str), rpm=meta_jnu["rpm"].values,
             scheme=v)
del sig, jnu
print("JNU pronto |", tempo())


# %% ===================================================================== P
# PRONOSTIA, os 17 rolamentos. Leitores verbatim da rodada 4.

!git clone -q https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset.git
BASE = "ieee-phm-2012-data-challenge-dataset"
PRON_COND = {1: (1800, 4000), 2: (1650, 4200), 3: (1500, 5000)}   # rpm, N


def list_bearing_files(bearing_dir_, prefix="acc_"):
    fs_ = [f for f in os.listdir(bearing_dir_)
           if f.startswith(prefix) and f.endswith(".csv")]
    fs_ = sorted(fs_, key=lambda s: int(re.findall(r"\d+", s)[0]))
    return [os.path.join(bearing_dir_, f) for f in fs_]


def read_pronostia_df(path):
    df = pd.read_csv(path, header=None, sep=",", engine="python")
    if df.shape[1] < 5:
        df = pd.read_csv(path, header=None, sep=";", engine="python")
    return df


def pronostia_input_antigo(df):
    """Esquema das rodadas 2-4 (2 reais + 4 zeros). SO para a fidelidade."""
    acc = df.iloc[:, -2:].to_numpy(dtype=np.float32)
    acc = (acc - acc.mean(axis=0)) / (acc.std(axis=0) + 1e-8)
    x = np.zeros((6, acc.shape[0]), dtype=np.float32)
    x[0, :], x[1, :] = acc[:, 0], acc[:, 1]
    return x


def pronostia_rep_snap(df):
    acc = df.iloc[:, -2:].to_numpy(dtype=np.float32)
    return fill_cyclic(windowize(znorm_snap(acc).T))


def pronostia_rep_win(df):
    acc = df.iloc[:, -2:].to_numpy(dtype=np.float32)
    return fill_cyclic(znorm_win(windowize(acc.T)))


def clock_seconds(df):
    h_, m_, s_, us_ = df.iloc[0, :4].astype(float)
    return h_ * 3600 + m_ * 60 + s_ + us_ * 1e-6


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


LOC = {}
for sub in sorted(os.listdir(BASE)):
    p_ = os.path.join(BASE, sub)
    if not os.path.isdir(p_) or sub.startswith("."):
        continue
    for b in sorted(os.listdir(p_)):
        if b.startswith("Bearing") and os.path.isdir(os.path.join(p_, b)):
            LOC.setdefault(b, {})[sub] = len(list_bearing_files(os.path.join(p_, b)))
PREFERENCIA = ("Full_Test_Set", "Learning_set", "Test_set")
PR_OUT = os.path.join(SUB["A"], "pronostia")
os.makedirs(PR_OUT, exist_ok=True)


def run_pronostia(bearing):
    subs = LOC.get(bearing, {})
    sub = next((s for s in PREFERENCIA if s in subs), None)
    if sub is None:
        print("%s: nao encontrado" % bearing)
        return
    files = list_bearing_files(os.path.join(BASE, sub, bearing))
    cond = int(re.match(r"Bearing(\d)_", bearing).group(1))
    n_trunc = -1
    if sub == "Full_Test_Set" and "Test_set" in subs:
        tfiles = list_bearing_files(os.path.join(BASE, "Test_set", bearing))
        if (len(tfiles) <= len(files) and md5(tfiles[0]) == md5(files[0])
                and md5(tfiles[-1]) == md5(files[len(tfiles) - 1])):
            n_trunc = len(tfiles)
    fid = bearing == "Bearing1_1"
    embs = {"rep_snap": [], "rep_win": []}
    nrms = {"rep_snap": [], "rep_win": []}
    old, t0s = [], []
    for f in files:
        df = read_pronostia_df(f)
        t0s.append(clock_seconds(df))
        for v, prep in (("rep_snap", pronostia_rep_snap), ("rep_win", pronostia_rep_win)):
            e, n = embed_windows_array(prep(df), with_norm=True)
            embs[v].append(e.astype(EMB_DTYPE))
            nrms[v].append(n)
        if fid:
            old.append(embed_windows_array(windowize(pronostia_input_antigo(df))))
    names = np.array([os.path.basename(f) for f in files])
    print("%s: cond %d, %s, %d snapshots%s | %s" % (
        bearing, cond, sub, len(files),
        "  (truncado no desafio: %d)" % n_trunc if n_trunc > 0 else "", tempo()))
    if fid:
        n_leg = max(5, int(ENROL_FRAC_LEGADO * len(files)))
        mu, invC, thr = legacy_model(np.vstack(old[:n_leg]))
        compare_to_canon("pronostia_Bearing1_1", pct_series(old, mu, invC, thr),
                         "pronostia_bearing1_1.csv")
    for v, e in embs.items():
        save_emb(PR_OUT, "pronostia_%s_%s" % (v, bearing), e, z_norm=nrms[v],
                 files=names, subset=sub,
                 n_truncated=n_trunc, t0_seconds=np.array(t0s), condition=cond, scheme=v)


for b in sorted(LOC, key=lambda s: tuple(int(x) for x in re.findall(r"\d+", s))):
    run_pronostia(b)


# %% ===================================================================== I1
# IMS: download e contagem de colunas. Igual as rodadas anteriores.

!wget -q -nc https://data.nasa.gov/docs/legacy/IMS.zip
!unzip -q -o IMS.zip -d IMS_DATA
!apt-get -qq update && apt-get -qq install -y unrar > /dev/null

for rar in sorted(glob.glob("IMS_DATA/IMS/*.rar")):
    print("extraindo", os.path.basename(rar))
    os.system('unrar x -o+ -inul "%s" IMS_DATA/IMS/' % rar)


def pasta_com_dados(raiz):
    for _ in range(3):
        itens = [i for i in os.listdir(raiz) if not i.startswith(".")]
        arqs = [i for i in itens if os.path.isfile(os.path.join(raiz, i))]
        subs = [i for i in itens if os.path.isdir(os.path.join(raiz, i))]
        if arqs:
            return raiz
        if len(subs) == 1:
            raiz = os.path.join(raiz, subs[0])
        else:
            break
    return raiz


IMS_NCOL = {}
for seq in sorted(d for d in os.listdir("IMS_DATA/IMS")
                  if os.path.isdir(os.path.join("IMS_DATA/IMS", d))):
    folder = pasta_com_dados(os.path.join("IMS_DATA/IMS", seq))
    files = sorted(f for f in os.listdir(folder)
                   if not f.startswith(".") and os.path.isfile(os.path.join(folder, f)))
    if not files:
        continue
    ncols = pd.read_csv(os.path.join(folder, files[0]), sep=r"\s+", header=None).shape[1]
    IMS_NCOL[seq] = (folder, ncols, len(files))
    print("%-10s %5d arquivos  %d colunas" % (seq, len(files), ncols))

# um detector por rolamento, colunas conferidas na rodada 3
IMS_BEARINGS = {
    "1st_test": {"b1": [0, 1], "b2": [2, 3], "b3": [4, 5], "b4": [6, 7]},
    "2nd_test": {"b1": [0], "b2": [1], "b3": [2], "b4": [3]},
    "4th_test": {"b1": [0], "b2": [1], "b3": [2], "b4": [3]},
}
for s, vs in IMS_BEARINGS.items():
    assert s in IMS_NCOL, "sequencia %s nao encontrada" % s
    for v, cols in vs.items():
        assert max(cols) < IMS_NCOL[s][1], "%s/%s pede coluna %d" % (s, v, max(cols))
print("rolamentos conferidos contra as colunas reais")


def ims_input_antigo(raw, cols):
    """Esquema das rodadas 2-3 (reais + zeros, snap). SO para a fidelidade."""
    data = raw[:, cols]
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    padded = np.zeros((data.shape[0], 6), dtype=np.float32)
    padded[:, :data.shape[1]] = data
    return padded.T


def ims_rep_snap(raw, cols):
    return fill_cyclic(windowize(znorm_snap(raw[:, cols]).T))


def ims_rep_win(raw, cols):
    return fill_cyclic(znorm_win(windowize(np.ascontiguousarray(raw[:, cols].T))))


# %% ===================================================================== I2
# IMS: embeddings dos 12 rolamentos nas duas variantes. Ordem 2nd -> 1st -> 4th:
# a fidelidade (2nd_test, 4 canais, esquema antigo) sai logo no inicio.

def run_ims(seq):
    folder, ncols, _ = IMS_NCOL[seq]
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if not f.startswith(".") and os.path.isfile(os.path.join(folder, f)))
    embs = {(v, b): [] for v in IMS_VARIANTES for b in IMS_BEARINGS[seq]}
    nrms = {(v, b): [] for v in IMS_VARIANTES for b in IMS_BEARINGS[seq]}
    fid = seq == "2nd_test"
    old = []
    t0 = time.time()
    for i, f in enumerate(files):
        if i and i % 500 == 0:
            el = time.time() - t0
            print("   %s %5d/%d  %.0f s, ~%.0f s restantes"
                  % (seq, i, len(files), el, el * (len(files) - i) / i), flush=True)
        raw = pd.read_csv(f, sep=r"\s+", header=None).values.astype(np.float32)
        for b, cols in IMS_BEARINGS[seq].items():
            for v, prep in (("rep_snap", ims_rep_snap), ("rep_win", ims_rep_win)):
                if v not in IMS_VARIANTES:
                    continue
                e, n = embed_windows_array(prep(raw, cols), with_norm=True)
                embs[(v, b)].append(e.astype(EMB_DTYPE))
                nrms[(v, b)].append(n)
        if fid:
            old.append(embed_windows_array(windowize(ims_input_antigo(raw, [0, 1, 2, 3]))))
    names = np.array([os.path.basename(f) for f in files])
    print("%s: %d snapshots | %s" % (seq, len(files), tempo()))
    if fid:
        n_leg = int(ENROL_FRAC_LEGADO * len(files))
        mu, invC, thr = legacy_model(np.vstack(old[:n_leg]))
        compare_to_canon("ims_%s" % seq, pct_series(old, mu, invC, thr), "ims_%s.csv" % seq)
        del old
    for (v, b), e in embs.items():
        save_emb(SUB["ims_" + v], "ims_%s_%s" % (seq, b), e, z_norm=nrms[(v, b)], files=names,
                 channels=np.array(IMS_BEARINGS[seq][b]), n_columns=ncols, scheme=v)
    del embs, nrms


for seq in ("2nd_test", "1st_test", "4th_test"):
    run_ims(seq)


# %% ===================================================================== G
# Manifesto, zips, copia para o Drive e download.

manifest = dict(
    checkpoint_sha=CKPT_SHA, checkpoint_arquivo=os.path.basename(CKPT),
    checkpoint_sha_rodada5=CKPT_SHA_RODADA5, ims_variantes=list(IMS_VARIANTES),
    rodada=9, seed=SEED, window=WINDOW_SIZE, stride=STRIDE,
    arquitetura=dict(pooling="MaxPool1d(k=%d, s=%d, p=%d) apos cada ReLU"
                     % (POOL_K, POOL_S, POOL_P),
                     comprimentos="1024 -> 512 -> 256 -> 128",
                     campo_receptivo=int(_rf)),
    z_norm="||z~|| antes da eq. (4), por janela, guardado nos .npz",
    emb_dtype=str(np.dtype(EMB_DTYPE)),
    regra_canais="repeticao ciclica: posicao j recebe o canal j mod k",
    normalizacoes=dict(snap="z-score por canal no snapshot (IMS, PRONOSTIA)",
                       rec="z-score por canal na gravacao inteira (JNU, MAFAULDA)",
                       win="z-score por canal em cada janela (treino da CNN)"),
    ims_columns={k: v[1] for k, v in IMS_NCOL.items()}, ims_bearings=IMS_BEARINGS,
    pronostia_locations=LOC, pronostia_conditions=PRON_COND,
    fidelidade=FIDELIDADE, minutos=round((time.time() - T_START) / 60, 1),
    files={k: sorted(os.path.relpath(os.path.join(dp, f), d)
                     for dp, _, fs in os.walk(d) for f in fs) for k, d in SUB.items()},
)
json.dump(manifest, open(os.path.join(SUB["A"], "manifest.json"), "w"), indent=2)
print(json.dumps(FIDELIDADE, indent=2))
# Rodada 6: os CSVs canonicos vieram do checkpoint antigo, entao divergir e o
# esperado. O numero fica registrado como medida de quanto o retreino mexeu em
# cada base, e nao reprova nada.
print("\ncomparacao com a rodada 5 (esperado DIVERGIR -- o checkpoint mudou):")
for k, v in sorted(FIDELIDADE.items()):
    print("   %-26s max|dif| = %8.4f pp" % (k, v.get("max_dif", float("nan"))))

ZIPS = []
for k, d in SUB.items():
    z = "/content/rerun9_%s.zip" % k
    os.system('cd %s && zip -q -r -0 %s %s' % (ROOT, z, k))
    ZIPS.append(z)
    print("zip %-32s %7.1f MB" % (os.path.basename(z), os.path.getsize(z) / 1e6))

# [RODADA 6] o checkpoint agora fica na raiz do Drive, entao CANON_DIR e o
# proprio MyDrive: copiar para o pai dele cairia em /content/drive, que nao
# sincroniza. Grava sempre dentro de MyDrive.
DRIVE = "/content/drive/MyDrive"
if os.path.isdir(DRIVE):
    for z in ZIPS:
        destino = os.path.join(DRIVE, os.path.basename(z))
        shutil.copy(z, destino)
        print("copiado para o Drive:", destino)
else:
    print("Drive nao montado: pegue os zips pelo download abaixo.")

try:
    from google.colab import files as colab_files
    for z in ZIPS:
        colab_files.download(z)
except Exception as e:
    print("download automatico indisponivel (%s); pegue pelo Drive." % e)
print("fim |", tempo())
