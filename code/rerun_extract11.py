#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 11 -- IMS, PRONOSTIA e JNU com as representacoes de comparacao.

Este .py e a FONTE (gerado por gera_rodada11.py a partir do rerun_extract10). O que se abre no
Colab e rerun_extract11.ipynb, gerado por `python make_notebook6.py rerun_extract11`.

O QUE FAZ
---------
Cada arquivo e lido UMA vez e codificado em oito representacoes, por janela:
    r9      a rede do artigo (embedding_model_r9_seed42.pth); serve de FIDELIDADE, porque
            localmente tem de coincidir com code/rerun9/
    ce      a mesma rede treinada como classificador normal x desbalanceamento
    rnd0-1  a mesma rede sem treino, dois sorteios de pesos
    zer     a rede r9 com zeros nas posicoes sem canal real, no lugar da repeticao em ciclo
    feat    indicadores classicos do sinal cru (RMS, pico, curtose, assimetria, crista)
    spec    espectro do sinal cru em 16 bandas
Os leitores, o janelamento, a regra de canais e a deteccao dos arquivos truncados do PRONOSTIA
sao os do rerun_extract9, sem mudanca. O protocolo do artigo roda depois, localmente, sobre cada
representacao: so ela muda.

ORDEM DOS NOTEBOOKS
-------------------
    1. train_rodada11.ipynb    deixa embedding_model_r11_ce.pth, _rnd0.pth e _rnd1.pth na raiz
                               do Drive
    2. rerun_extract11.ipynb   este; precisa tambem do embedding_model_r9_seed42.pth no Drive
    3. analise local           descompactar os zips em code/rerun11/

SAIDA
-----
Uma pasta por representacao, com os mesmos nomes de arquivo de code/rerun9/:
    <tag>/A/jnu/jnu_rep_win_emb.npz
    <tag>/A/pronostia/pronostia_rep_win_BearingX_Y_emb.npz
    <tag>/ims_rep_win/ims_<seq>_b<k>_emb.npz
e um zip por representacao, rerun11_<tag>.zip (~350 MB os de rede, dezenas de MB os outros).

TEMPO
-----
GPU T4, ~1 h 15: a leitura dos arquivos domina.
"""

# %% ===================================================================== E0
# Setup. Nada aqui treina.

import os, re, json, glob, time, hashlib, shutil, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TESTE = os.environ.get("RERUN11_TESTE") == "1"      # ensaio local com dados falsos

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cpu" and not TESTE:
    print("AVISO: runtime em CPU. Ambiente de execucao > Alterar tipo > GPU T4.")

WINDOW_SIZE = 1024
STRIDE = 512
EMB_DTYPE = np.float16
POOL_K, POOL_S, POOL_P = 4, 2, 1
N_RND = 2

ROOT = os.environ["RERUN11_ROOT"] if TESTE else "/content/rerun11"
DADOS = os.environ["RERUN11_DADOS"] if TESTE else "/content"
os.makedirs(ROOT, exist_ok=True)
T_START = time.time()


def tempo():
    return "%.1f min" % ((time.time() - T_START) / 60)


try:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception as e:
    print("Drive nao montado (%s). Seguindo so com %s." % (e, DADOS))


def sh(cmd):
    """Comando de shell; no ensaio local, nada e baixado."""
    if TESTE:
        print("   (ensaio) pulado:", cmd)
        return
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=False, cwd=DADOS)


# %% ===================================================================== E1
# Rede e checkpoints. A rede e a das rodadas 7 e 9.

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

    def forward(self, x):
        return F.normalize(self.head(self.feat(x.float()).squeeze(-1)), dim=1)


def procurar(nome, raizes=(DADOS, "/content/drive/MyDrive"), max_dirs=40000):
    """Primeiro caminho encontrado para `nome`, podando pastas de sistema."""
    IGNORAR = {".git", ".ipynb_checkpoints", "__pycache__", "sample_data", "IMS_DATA",
               "JNU-Bearing-Dataset", "ieee-phm-2012-data-challenge-dataset",
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


def hash_pesos(state):
    """O mesmo hash do state_dict que o rerun_extract9 grava em `ckpt`."""
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


NOMES = {"r9": "embedding_model_r9_seed42.pth", "ce": "embedding_model_r11_ce.pth"}
NOMES.update({"rnd%d" % s: "embedding_model_r11_rnd%d.pth" % s for s in range(N_RND)})
MODELOS, CKPT = {}, {}
for tag, nome in NOMES.items():
    p = procurar(nome)
    if p is None:
        print("   %-3s %-32s NAO ENCONTRADO" % (tag, nome))
        continue
    state = torch.load(p, map_location=DEVICE, weights_only=True)
    net = EmbeddingNet1D().to(DEVICE)
    net.load_state_dict(state, strict=True)
    net.eval()
    MODELOS[tag] = net
    CKPT[tag] = dict(arquivo=p, pesos=hash_pesos(state))
    print("   %-3s %-32s pesos %s" % (tag, nome, CKPT[tag]["pesos"]))
faltam = [t for t in NOMES if t not in MODELOS]
if faltam:
    raise SystemExit("faltam os checkpoints %s; o r9 e o da rodada 9, os outros saem do train_rodada11" % faltam)

# representacoes sem rede propria: a rede r9 com zeros no lugar da repeticao de canais ("zer"),
# os indicadores classicos ("feat") e o espectro em bandas ("spec")
TAGS = list(MODELOS) + ["zer", "feat", "spec"]
CKPT["zer"] = dict(arquivo=CKPT["r9"]["arquivo"], pesos=CKPT["r9"]["pesos"])
CKPT["feat"] = dict(arquivo="", pesos="indicadores classicos")
CKPT["spec"] = dict(arquivo="", pesos="espectro em bandas")
ESQUEMA = {t: "rep_win" for t in MODELOS}
ESQUEMA.update(zer="zero_win", feat="cru", spec="cru")

with torch.no_grad():
    _t = torch.zeros(1, 6, WINDOW_SIZE).to(DEVICE)
    _comp = [_t.shape[-1]]
    for _c in next(iter(MODELOS.values())).feat:
        _t = _c(_t)
        if isinstance(_c, nn.MaxPool1d):
            _comp.append(_t.shape[-1])
    assert _comp == [1024, 512, 256, 128], "geometria do pooling: %s" % _comp

SUB = {t: {k: os.path.join(ROOT, t, *k.split("/")) for k in ("A/jnu", "A/pronostia", "ims_rep_win")}
       for t in TAGS}
for t in SUB.values():
    for d in t.values():
        os.makedirs(d, exist_ok=True)


# %% ===================================================================== E2
# Funcoes. Janelamento, normalizacao e regra de canais identicas ao rerun_extract9.

N_ESPEC = 16


def feats_classicas(Xraw):
    """(N, k, L) cru -> (N, 5 k) float32. Indicadores classicos de cada janela, por canal: log10 do
    RMS, log10 do pico, curtose, assimetria e fator de crista."""
    X = np.asarray(Xraw, np.float64)
    d = X - X.mean(axis=2, keepdims=True)
    sd = np.sqrt((d ** 2).mean(axis=2)) + 1e-12
    rms = np.sqrt((X ** 2).mean(axis=2)) + 1e-12
    pico = np.abs(X).max(axis=2) + 1e-12
    cols = [np.log10(rms), np.log10(pico), (d ** 4).mean(axis=2) / sd ** 4,
            (d ** 3).mean(axis=2) / sd ** 3, pico / rms]
    return np.concatenate(cols, axis=1).astype(np.float32)


def espectro_bandas(Xraw):
    """(N, k, L) cru -> (N, 16 k) float32: log10 da energia em 16 bandas iguais de 0 a fs/2
    (janela de Hann, sem a componente continua)."""
    X = np.asarray(Xraw, np.float64)
    N, k, L = X.shape
    d = X - X.mean(axis=2, keepdims=True)
    P = np.abs(np.fft.rfft(d * np.hanning(L), axis=2)) ** 2
    nb = (P.shape[2] - 1) // N_ESPEC
    bandas = P[:, :, 1:1 + nb * N_ESPEC].reshape(N, k, N_ESPEC, nb).sum(axis=3)
    return np.log10(bandas + 1e-30).reshape(N, k * N_ESPEC).astype(np.float32)


def fill_zeros(Xw):
    """(N, k, L) -> (N, 6, L); as posicoes sem canal real ficam em zero."""
    out = np.zeros((Xw.shape[0], 6, Xw.shape[2]), Xw.dtype)
    out[:, :Xw.shape[1], :] = Xw
    return out


@torch.no_grad()
def codifica(Xraw, batch=256):
    """Janelas cruas (N, k, L) -> {tag: matriz por janela}. As redes recebem o que o artigo usa
    (z-score por janela, canais repetidos em ciclo); "zer" e a rede r9 com zeros no lugar da
    repeticao; "feat" e "spec" saem do sinal cru."""
    assert Xraw is not None and len(Xraw) > 0, "aquisicao sem janelas"
    Z = znorm_win(Xraw)
    Xc, Xz = fill_cyclic(Z), fill_zeros(Z)
    out = {t: [] for t in MODELOS}
    out["zer"] = []
    for i in range(0, len(Xc), batch):
        xb = torch.tensor(Xc[i:i + batch]).to(DEVICE)
        for t, net in MODELOS.items():
            out[t].append(net(xb).cpu().numpy())
        out["zer"].append(MODELOS["r9"](torch.tensor(Xz[i:i + batch]).to(DEVICE)).cpu().numpy())
    res = {t: np.vstack(v).astype(EMB_DTYPE) for t, v in out.items()}
    res["feat"] = feats_classicas(Xraw)
    res["spec"] = espectro_bandas(Xraw).astype(EMB_DTYPE)
    return res


def windowize(x, window_size=WINDOW_SIZE, stride=STRIDE):
    """x: (C, T) -> (N, C, window_size)"""
    C, T = x.shape
    wins = [x[:, s:s + window_size] for s in range(0, T - window_size + 1, stride)]
    return np.stack(wins, axis=0) if wins else np.empty((0, C, window_size), np.float32)


def znorm_win(Xw):
    """(N, C, L) -> z-score por janela e canal."""
    mean = Xw.mean(axis=2, keepdims=True)
    std = Xw.std(axis=2, keepdims=True) + 1e-8
    return (Xw - mean) / std


def fill_cyclic(Xw):
    """(N, k, L) -> (N, 6, L); a posicao j recebe o canal j mod k."""
    k = Xw.shape[1]
    return np.ascontiguousarray(Xw[:, [j % k for j in range(6)], :])


def salva(sub, nome, por_tag, **meta):
    """por_tag: {tag: lista de matrizes (janelas x 128), uma por gravacao ou snapshot}."""
    for t, embs in por_tag.items():
        offsets = np.cumsum([0] + [len(E) for E in embs]).astype(np.int64)
        path = os.path.join(SUB[t][sub], nome + "_emb.npz")
        np.savez_compressed(path, E=np.concatenate(embs), offsets=offsets,
                            ckpt=CKPT[t]["pesos"], window=WINDOW_SIZE, stride=STRIDE,
                            scheme=ESQUEMA[t], **meta)
    print("   salvo %-40s %8d janelas x %d representacoes" % (nome, offsets[-1], len(por_tag)))


# %% ===================================================================== J
# JNU. Leitores do rerun_extract9.

sh("git clone -q https://github.com/ClarkGableWang/JNU-Bearing-Dataset.git")


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


LAB = {"n": "normal", "ib": "inner", "ob": "outer", "tb": "ball"}
meta_jnu = []
for p in sorted(glob.glob(os.path.join(DADOS, "JNU-Bearing-Dataset", "**", "*.csv"), recursive=True)):
    b = os.path.basename(p)
    m = re.match(r"([a-z]+)(600|800|1000)", b.lower())
    if m and m.group(1) in LAB:
        meta_jnu.append(dict(path=p, label=LAB[m.group(1)], rpm=int(m.group(2))))
meta_jnu = pd.DataFrame(meta_jnu)
print(meta_jnu.groupby(["rpm", "label"]).size())

jnu = {t: [] for t in TAGS}
for _i, p in enumerate(meta_jnu["path"]):
    _x = read_jnu_csv_1ch(p)
    _cru = windowize(_x[None, :])
    if _i == 0:        # o caminho novo tem de dar a mesma entrada de rede do rerun_extract9
        assert np.allclose(fill_cyclic(znorm_win(_cru)), jnu_rep_win(_x), atol=1e-6), "JNU: entrada difere"
    for t, E in codifica(_cru).items():
        jnu[t].append(E)
salva("A/jnu", "jnu_rep_win", jnu,
      files=np.array([os.path.basename(p) for p in meta_jnu["path"]]),
      labels=meta_jnu["label"].values.astype(str), rpm=meta_jnu["rpm"].values)
del jnu
print("JNU pronto |", tempo())


# %% ===================================================================== P
# PRONOSTIA, os 17 rolamentos. Leitores e deteccao dos truncados do rerun_extract9.

sh("git clone -q https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset.git")
BASE = os.path.join(DADOS, "ieee-phm-2012-data-challenge-dataset")
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
    embs = {t: [] for t in TAGS}
    t0s = []
    for _i, f in enumerate(files):
        df = read_pronostia_df(f)
        t0s.append(clock_seconds(df))
        _cru = windowize(df.iloc[:, -2:].to_numpy(dtype=np.float32).T)
        if _i == 0:
            assert np.allclose(fill_cyclic(znorm_win(_cru)), pronostia_rep_win(df), atol=1e-6), "PRONOSTIA: entrada difere"
        for t, E in codifica(_cru).items():
            embs[t].append(E)
    print("%s: cond %d, %s, %d snapshots%s | %s" % (
        bearing, cond, sub, len(files),
        "  (truncado no desafio: %d)" % n_trunc if n_trunc > 0 else "", tempo()))
    salva("A/pronostia", "pronostia_rep_win_%s" % bearing, embs,
          files=np.array([os.path.basename(f) for f in files]), subset=sub,
          n_truncated=n_trunc, t0_seconds=np.array(t0s), condition=cond)


for b in sorted(LOC, key=lambda s: tuple(int(x) for x in re.findall(r"\d+", s))):
    run_pronostia(b)
print("PRONOSTIA pronto |", tempo())


# %% ===================================================================== I1
# IMS: download e contagem de colunas. Igual ao rerun_extract9.

sh("wget -q -nc https://data.nasa.gov/docs/legacy/IMS.zip")
sh("unzip -q -o IMS.zip -d IMS_DATA")
sh("apt-get -qq update && apt-get -qq install -y unrar > /dev/null")
IMS_DIR = os.path.join(DADOS, "IMS_DATA", "IMS")
for rar in sorted(glob.glob(os.path.join(IMS_DIR, "*.rar"))):
    print("extraindo", os.path.basename(rar))
    os.system('unrar x -o+ -inul "%s" "%s/"' % (rar, IMS_DIR))


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
for seq in sorted(d for d in os.listdir(IMS_DIR) if os.path.isdir(os.path.join(IMS_DIR, d))):
    folder = pasta_com_dados(os.path.join(IMS_DIR, seq))
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


def ims_rep_win(raw, cols):
    return fill_cyclic(znorm_win(windowize(np.ascontiguousarray(raw[:, cols].T))))


# %% ===================================================================== I2
# IMS: embeddings dos 12 rolamentos, na ordem 2nd -> 1st -> 4th do rerun_extract9.

def run_ims(seq):
    folder, ncols, _ = IMS_NCOL[seq]
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if not f.startswith(".") and os.path.isfile(os.path.join(folder, f)))
    embs = {(t, b): [] for t in TAGS for b in IMS_BEARINGS[seq]}
    t0 = time.time()
    for i, f in enumerate(files):
        if i and i % 500 == 0:
            el = time.time() - t0
            print("   %s %5d/%d  %.0f s, ~%.0f s restantes"
                  % (seq, i, len(files), el, el * (len(files) - i) / i), flush=True)
        raw = pd.read_csv(f, sep=r"\s+", header=None).values.astype(np.float32)
        for b, cols in IMS_BEARINGS[seq].items():
            _cru = windowize(np.ascontiguousarray(raw[:, cols].T))
            if i == 0:
                assert np.allclose(fill_cyclic(znorm_win(_cru)), ims_rep_win(raw, cols), atol=1e-6), "IMS: entrada difere"
            for t, E in codifica(_cru).items():
                embs[(t, b)].append(E)
    names = np.array([os.path.basename(f) for f in files])
    print("%s: %d snapshots | %s" % (seq, len(files), tempo()))
    for b in IMS_BEARINGS[seq]:
        salva("ims_rep_win", "ims_%s_%s" % (seq, b), {t: embs[(t, b)] for t in TAGS},
              files=names, channels=np.array(IMS_BEARINGS[seq][b]), n_columns=ncols)
    del embs


for seq in ("2nd_test", "1st_test", "4th_test"):
    run_ims(seq)


# %% ===================================================================== G
# Manifesto, zips, copia para o Drive e download.

manifest = dict(
    rodada=11, checkpoints=CKPT, esquema=ESQUEMA,
    window=WINDOW_SIZE, stride=STRIDE, emb_dtype=str(np.dtype(EMB_DTYPE)),
    ims_columns={k: v[1] for k, v in IMS_NCOL.items()}, ims_bearings=IMS_BEARINGS,
    pronostia_locations=LOC, pronostia_conditions=PRON_COND, teste_local=TESTE,
    minutos=round((time.time() - T_START) / 60, 1),
    files={t: sorted(os.path.relpath(os.path.join(dp, f), os.path.join(ROOT, t))
                     for dp, _, fs in os.walk(os.path.join(ROOT, t)) for f in fs) for t in TAGS},
)
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
for t in TAGS:
    shutil.copy(os.path.join(ROOT, "manifest.json"), os.path.join(ROOT, t, "manifest.json"))
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2)[:3000])

if not TESTE:
    ZIPS = []
    for t in TAGS:
        z = "/content/rerun11_%s.zip" % t
        os.system("cd %s && zip -q -r -0 %s %s" % (ROOT, z, t))
        ZIPS.append(z)
        print("zip %-24s %7.1f MB" % (os.path.basename(z), os.path.getsize(z) / 1e6))
    DRIVE = "/content/drive/MyDrive"
    if os.path.isdir(DRIVE):
        for z in ZIPS:
            shutil.copy(z, os.path.join(DRIVE, os.path.basename(z)))
            print("copiado para o Drive:", os.path.join(DRIVE, os.path.basename(z)))
    else:
        print("Drive nao montado: pegue os zips pelo download abaixo.")
    try:
        from google.colab import files as colab_files
        for z in ZIPS:
            colab_files.download(z)
    except Exception as e:
        print("download automatico indisponivel (%s); pegue pelo Drive." % e)
print("fim |", tempo())
