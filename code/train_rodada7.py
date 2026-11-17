#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 7 -- max pooling entre as convolucoes, treino binario da rodada 5.

Este .py e a FONTE. O que se abre no Colab e train_rodada7.ipynb, gerado por
`python make_notebook6.py train_rodada7`.

POR QUE
-------
Medimos (achados-da-rerodada.md, secao 29) que a arquitetura das rodadas 1-6
nao tem downsampling nenhum entre as convolucoes: o tensor vai de 6 x 1024 a
128 x 1024 e so colapsa no AdaptiveAvgPool. Com kernel 9, passo 1 e tres
camadas, o campo receptivo de cada feature e de apenas 25 amostras -- 1,25 ms
no IMS. Uma volta do eixo sao 33 Hz; uma feature nunca ve nem 1/25 de volta.
A rede e um detector de textura local seguido de media global, e nao alcanca o
periodo de repeticao dos impulsos, que e o que define BPFO e BPFI.

O QUE MUDA
----------
1. MAX POOLING depois de cada bloco conv+BN+ReLU, com kernel 4 e passo 2
   (padding 1), que reduz o comprimento pela metade:
       1024 -> 512 -> 256 -> 128
   O kernel 4 com passo 2 e sobreposto, como no AlexNet. E a unica geometria
   que satisfaz as duas condicoes pedidas: kernel 4 E a sequencia acima.
   Campo receptivo: 25 -> 78 amostras, 3,1 vezes maior.

       camada   RF   salto          IMS (20 kHz)   fs/RF
       conv1     9     1
       pool1    12     2
       conv2    28     2
       pool2    34     4
       conv3    66     4
       pool3    78     8            3,90 ms        256 Hz   (era 1,25 ms / 800 Hz)

2. TREINO BINARIO DA RODADA 5, nao o multiclasse da rodada 6: so normal e
   desbalanceamento entram nos trios, com ancora e positivo sendo duas janelas
   normais da MESMA rotacao e o negativo uma janela de desbalanceamento da
   mesma rotacao. A rodada 6 mostrou que multiplicar classes dentro da mesma
   bancada melhora o MAFAULDA e nao transfere; volta-se ao treino que sustenta
   a contribuicao de open-set.

3. CURVA DE PERDA desenhada por epoca no proprio notebook.

O QUE NAO MUDA
--------------
Janela 1024, passo 512, z-score por janela, kernel 9 nas convolucoes, canais
6->32->64->128, ReLU depois de cada BN, saida 128-d normalizada em l2,
TripletMarginLoss(margin=0.8, p=2), Adam lr=3e-4, 12 epocas, 2000 passos por
epoca, lote 64, aug_weak identica. O detector segue de uma classe so.

PARTICAO
--------
Mantida a da rodada 6 -- falhas por gravacao, normais contiguas no tempo --,
que corrige a limitacao declarada no artigo. Ela nao confunde a comparacao:
as quatro familias nao treinadas continuam sendo open-set, e as tres bases
externas, que sao o que decide, nao dependem da particao do MAFAULDA.

O QUE SALVA
-----------
  rodada7_treino.zip
    embedding_model_r7_seed<S>.pth      checkpoint por semente, com o sha
    treino_log.csv                      perda por epoca e por semente
    curva_perda.png                     a figura da perda
    split_gravacoes.json                a particao, auditavel
    mafaulda_avaliacao.csv              pct por gravacao, com rotulos
    mafaulda_metricas.json              metricas, com recorte open-set
    mafaulda_emb/*.npz                  embeddings rotulados por gravacao
    mafaulda_feats.npz                  features de amplitude
    manifest.json

DEPOIS
------
Rodar rerun_extract7.ipynb, que pega embedding_model_r7_seed42.pth no Drive e
reextrai IMS, PRONOSTIA e JNU.

TEMPO
-----
GPU T4, uma semente: ~1 h 30. O MAFAULDA e lido uma vez so em R5 e uma vez em
R8, como na rodada 6.
"""

# %% ===================================================================== R1
# Setup. Nada aqui treina.

import os, re, json, glob, time, random, hashlib, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TripletMarginLoss
from torch.optim import Adam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

WINDOW_SIZE = 1024
STRIDE = 512
LAMBDA = 1e-3
QUANT = 0.99
DECISION_LEVEL = 50.0                # pct de janelas anomalas que decide a gravacao

# treino: identico ao original, exceto onde o docstring diz
MARGIN = 0.8
LR = 3e-4
EPOCHS = 12
STEPS_PER_EPOCH = 2000
BATCH = 64
SEEDS = (42,)                        # acrescente (42, 43, 44) para medir a variacao

# Quantas janelas de cada gravacao ficam em RAM para o treino. Cada gravacao
# tem ~487 janelas; guardar todas seriam ~23 GB. Com ~1950 gravacoes, 48
# janelas por gravacao dao ~94 mil janelas distintas e ~1,1 GB em float16 --
# cabe no Colab com folga. Suba se houver memoria.
MAX_WIN_TRAIN = 48

# Quantas classes candidatas sortear para escolher o negativo. O negativo deve
# estar na rotacao mais proxima possivel do ancora; sortear so uma classe as
# vezes cai numa que nao tem rotacao proxima no treino, e o atalho da rotacao
# volta. Com 4 candidatas escolhe-se a de menor distancia em rotacao.
# Max pooling: kernel 4 com passo 2 e padding 1 reduz 1024 -> 512 -> 256 -> 128.
POOL_K, POOL_S, POOL_P = 4, 2, 1

# Familias que entram nos trios. As demais ficam de fora do treino e voltam a
# ser open-set na avaliacao, como na rodada 5.
FAM_TREINO = ("normal", "imbalance")

EMB_DTYPE = np.float16
ROOT = "/content/rodada7"
EMB_DIR = os.path.join(ROOT, "mafaulda_emb")
for d in (ROOT, EMB_DIR):
    os.makedirs(d, exist_ok=True)
T_START = time.time()


def tempo():
    return "%.1f min" % ((time.time() - T_START) / 60)


try:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception as e:
    print("Drive nao montado (%s). Seguindo so com /content." % e)


# %% ===================================================================== R2
# Rede. Identica as rodadas anteriores -- so os pesos serao outros.

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
        # ||z~|| e o unico grau de liberdade que a eq. (4) descarta. Guardamos para
        # poder testar, depois, se ele carrega informacao (nunca foi medido).
        return (zn, z.norm(dim=1)) if return_norm else zn


@torch.no_grad()
def embed_windows_array(Xw, net, batch=256, dev=DEVICE, with_norm=False):
    net.eval()
    out, nrm = [], []
    for i in range(0, len(Xw), batch):
        xb = torch.tensor(np.asarray(Xw[i:i + batch], dtype=np.float32)).to(dev)
        if with_norm:
            e, n = net(xb, return_norm=True)
            out.append(e.cpu().numpy()); nrm.append(n.cpu().numpy())
        else:
            out.append(net(xb).cpu().numpy())
    E = np.vstack(out) if out else np.empty((0, 128), np.float32)
    if not with_norm:
        return E
    return E, (np.concatenate(nrm) if nrm else np.empty(0, np.float32))


def campo_receptivo():
    """RF e salto acumulados, camada a camada."""
    rf, salto, linhas = 1, 1, []
    for nome, k, st in (("conv1", 9, 1), ("pool1", POOL_K, POOL_S),
                        ("conv2", 9, 1), ("pool2", POOL_K, POOL_S),
                        ("conv3", 9, 1), ("pool3", POOL_K, POOL_S)):
        rf += (k - 1) * salto
        salto *= st
        linhas.append((nome, rf, salto))
    return rf, linhas


with torch.no_grad():
    _m = EmbeddingNet1D()
    _t = torch.zeros(1, 6, WINDOW_SIZE)
    _comp = [_t.shape[-1]]
    for _c in _m.feat:
        _t = _c(_t)
        if isinstance(_c, nn.MaxPool1d):
            _comp.append(_t.shape[-1])
    print("comprimento da serie: %s" % " -> ".join(map(str, _comp)))
    assert _comp == [1024, 512, 256, 128], "geometria do pooling saiu diferente: %s" % _comp
    _rf, _lin = campo_receptivo()
    print("campo receptivo por camada: %s" % ", ".join("%s %d" % (n, r) for n, r, _ in _lin))
    print("RF final %d amostras | IMS %.2f ms (fs 20 kHz) | MAFAULDA %.2f ms (fs 50 kHz)"
          % (_rf, 1000.0 * _rf / 20000, 1000.0 * _rf / 50000))
    print("parametros: {:,}".format(sum(p.numel() for p in _m.parameters())))
    del _m, _t


def sha_arquivo(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()[:16]


# %% ===================================================================== R3
# MAFAULDA: download e metadados. Agora com TODAS as familias, e a identidade
# de cada gravacao passa a ser o diretorio dela.

!pip -q install -U kagglehub
import kagglehub
try:
    MAF_PATH = kagglehub.dataset_download("vuxuancu/mafaulda-full")
except Exception as e:
    raise SystemExit("Download do MAFAULDA falhou (%s). Rode kagglehub.login() "
                     "com as credenciais do Kaggle e execute de novo." % e)
print("MAFAULDA em:", MAF_PATH, "|", tempo())


def extract_cond_from_filename(fname):
    """A rotacao esta no nome do arquivo: 12.288.csv -> 12.288 Hz."""
    m = re.search(r"(\d+(?:\.\d+)?)", fname)
    return float(m.group(1)) if m else None


def infer_fault_type(relpath):
    rp = relpath.replace("\\", "/").lower()
    if "/normal/" in "/%s/" % rp:
        return "normal"
    if "imbalance" in rp:
        return "imbalance"
    if "horizontal" in rp and "misalignment" in rp:
        return "horizontal-misalignment"
    if "vertical" in rp and "misalignment" in rp:
        return "vertical-misalignment"
    if "underhang" in rp:
        return "underhang-bearing"
    if "overhang" in rp:
        return "overhang-bearing"
    return "unknown"


rows = []
for root, _, files in os.walk(MAF_PATH):
    for f in files:
        if not f.lower().endswith(".csv"):
            continue
        full = os.path.join(root, f)
        rel = os.path.relpath(full, MAF_PATH).replace("\\", "/")
        partes = rel.split("/")
        rows.append(dict(
            filepath=full, relpath=rel,
            fault_type=infer_fault_type(rel),
            # a IDENTIDADE: o diretorio carrega tipo + local do defeito + severidade
            # normal | imbalance/6g | underhang/ball_fault/6g | ...
            label="/".join(partes[:-1]) or "normal",
            severity=partes[-2] if len(partes) >= 2 else None,
            cond=extract_cond_from_filename(f)))
meta = pd.DataFrame(rows).dropna(subset=["cond"])
meta = meta[meta.fault_type != "unknown"].reset_index(drop=True)

print("gravacoes: %d | familias: %d | classes (diretorios): %d | rotacoes: %d"
      % (len(meta), meta.fault_type.nunique(), meta.label.nunique(), meta.cond.nunique()))
print(meta.groupby("fault_type").agg(gravacoes=("relpath", "size"),
                                     classes=("label", "nunique"),
                                     rotacoes=("cond", "nunique")).to_string())

MAF_CH = ["underhang_axial", "underhang_radial", "underhang_tangential",
          "overhang_axial", "overhang_radial", "overhang_tangential"]
MAF_FS = 50000.0


# %% ===================================================================== R4
# Particao. Falhas por GRAVACAO (corrige a limitacao do artigo); normais
# CONTIGUAS NO TEMPO dentro da unica gravacao de cada rotacao.

SEED_SPLIT = 42
FRAC = (0.60, 0.20, 0.20)            # treino, enrolment/calibracao, teste

rng = np.random.default_rng(SEED_SPLIT)
split_rec = {}                       # relpath -> "train" | "cal" | "test"
for lab, g in meta[meta.fault_type != "normal"].groupby("label"):
    idx = g.relpath.tolist()
    rng.shuffle(idx)
    n = len(idx)
    n_tr, n_cal = int(FRAC[0] * n), int(FRAC[1] * n)
    if n >= 3 and n_tr + n_cal >= n:  # garante ao menos uma gravacao de teste
        n_tr, n_cal = max(1, n - 2), 1
    for i, r in enumerate(idx):
        split_rec[r] = "train" if i < n_tr else ("cal" if i < n_tr + n_cal else "test")

NORMAIS = meta[meta.fault_type == "normal"].relpath.tolist()
for r in NORMAIS:
    split_rec[r] = "normal-contiguo"

meta["split"] = meta.relpath.map(split_rec)
print("particao das falhas, por gravacao:")
print(meta[meta.fault_type != "normal"].groupby(["fault_type", "split"]).size()
      .unstack(fill_value=0).to_string())
print("\ngravacoes normais: %d, todas com particao contigua 60/20/20 no tempo"
      % len(NORMAIS))
json.dump({r: s for r, s in split_rec.items()},
          open(os.path.join(ROOT, "split_gravacoes.json"), "w"), indent=1)


# %% ===================================================================== R5
# Janelas. Mesma ordem do treino original: ler -> janelar -> z-score por
# janela -> (no treino) aug_weak. A aug vem DEPOIS da normalizacao, como no
# original, entao o fator de amplitude de fato atua.

def read_acc_only(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 8:
        raise ValueError("Arquivo com %d colunas, esperado >=8: %s" % (df.shape[1], csv_path))
    return df.iloc[:, 1:7]


def windows_norm(csv_path):
    """(n, 6, 1024) ja com z-score por janela e canal."""
    X = read_acc_only(csv_path).to_numpy(dtype=np.float32)
    wins = [X[s:s + WINDOW_SIZE, :].T
            for s in range(0, X.shape[0] - WINDOW_SIZE + 1, STRIDE)]
    if not wins:
        return np.empty((0, 6, WINDOW_SIZE), np.float32)
    Xw = np.stack(wins, axis=0)
    mean = Xw.mean(axis=2, keepdims=True)
    std = Xw.std(axis=2, keepdims=True) + 1e-8
    return (Xw - mean) / std


def blocos_contiguos(n, frac=FRAC):
    """Indices contiguos: nenhum par de janelas sobrepostas cruza a fronteira."""
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def aug_weak(x, max_shift=64, amp=(0.85, 1.15), noise_std=0.02):
    """Identica ao treino original."""
    C, T = x.shape
    y = np.roll(x, np.random.randint(-max_shift, max_shift + 1), axis=1)
    y = y * np.random.uniform(*amp)
    y = y + np.random.randn(C, T).astype(np.float32) * noise_std
    return y.astype(np.float32)


# Features de amplitude, calculadas NESTA MESMA passagem. Ler o MAFAULDA
# inteiro custa caro (~30 GB), entao cada gravacao e lida uma vez so aqui: o
# subconjunto de janelas de treino e as features saem da mesma leitura. As
# oito bandas de oitava sao as mesmas do IMS e do PRONOSTIA: de fs/256 a fs/2.

BAND_EDGES = np.array([0.0] + [MAF_FS / 2.0 / (2 ** k) for k in range(7, -1, -1)])
FEAT_NAMES = ["mean", "std", "rms", "peak", "p2p", "kurtosis", "skewness", "crest"] + \
             ["band_rms_%d" % i for i in range(8)]


def feats_gravacao(X):
    """(T, 6) -> (6, 16)."""
    out = np.zeros((X.shape[1], len(FEAT_NAMES)), np.float64)
    freqs = np.fft.rfftfreq(X.shape[0], d=1.0 / MAF_FS)
    faixa = [(freqs >= BAND_EDGES[i]) & (freqs < BAND_EDGES[i + 1]) for i in range(8)]
    for c in range(X.shape[1]):
        x = X[:, c].astype(np.float64)
        m, s = x.mean(), x.std()
        rms = np.sqrt(np.mean(x ** 2))
        pk = np.abs(x).max()
        z = (x - m) / (s + 1e-12)
        P = np.abs(np.fft.rfft(x)) ** 2 / len(x)
        bandas = [np.sqrt(P[f].sum() * 2 / len(x)) for f in faixa]
        out[c] = [m, s, rms, pk, x.max() - x.min(), (z ** 4).mean(), (z ** 3).mean(),
                  pk / (rms + 1e-12)] + bandas
    return out


print("passagem unica: janelas de treino (%s) + features de amplitude de tudo..."
      % " e ".join(FAM_TREINO), flush=True)
rng_w = np.random.default_rng(SEED_SPLIT)
TRAIN_W, TRAIN_META, F_all = {}, [], []
for i, r in enumerate(meta.itertuples()):
    X = read_acc_only(r.filepath).to_numpy(dtype=np.float32)
    F_all.append(feats_gravacao(X))
    if r.split in ("train", "normal-contiguo") and r.fault_type in FAM_TREINO:
        wins = [X[s:s + WINDOW_SIZE, :].T
                for s in range(0, X.shape[0] - WINDOW_SIZE + 1, STRIDE)]
        if wins:
            Xw = np.stack(wins, axis=0)
            Xw = (Xw - Xw.mean(axis=2, keepdims=True)) / (Xw.std(axis=2, keepdims=True) + 1e-8)
            pool = (np.arange(len(Xw)) if r.split == "train"
                    else blocos_contiguos(len(Xw))[0])
            if len(pool) >= 2:
                sel = (pool if len(pool) <= MAX_WIN_TRAIN
                       else rng_w.choice(pool, MAX_WIN_TRAIN, replace=False))
                TRAIN_W[r.relpath] = Xw[np.sort(sel)].astype(np.float16)
                TRAIN_META.append(dict(relpath=r.relpath, label=r.label, cond=r.cond,
                                       fault_type=r.fault_type))
    if (i + 1) % 200 == 0:
        print("   %4d/%4d gravacoes | cache %d | %s"
              % (i + 1, len(meta), len(TRAIN_W), tempo()), flush=True)

TRAIN_META = pd.DataFrame(TRAIN_META)
mb = sum(v.nbytes for v in TRAIN_W.values()) / 1e6
print("cache de treino: %d gravacoes, %d janelas, %.0f MB | %s"
      % (len(TRAIN_W), sum(len(v) for v in TRAIN_W.values()), mb, tempo()))

np.savez_compressed(os.path.join(ROOT, "mafaulda_feats.npz"),
                    feats=np.stack(F_all), feat_names=np.array(FEAT_NAMES),
                    relpath=meta.relpath.values.astype(str),
                    label=meta.label.values.astype(str),
                    fault_type=meta.fault_type.values.astype(str),
                    severity=meta.severity.values.astype(str),
                    cond=meta.cond.values.astype(float),
                    split=meta.split.values.astype(str),
                    band_edges_hz=BAND_EDGES, fs=MAF_FS, channels=np.array(MAF_CH))
del F_all
print("features salvas: %d gravacoes x 6 canais x %d | %s"
      % (len(meta), len(FEAT_NAMES), tempo()))


# %% ===================================================================== R6
# Trios. A mudanca de projeto da rodada 6:
#     positivo = mesma classe, ROTACAO DIFERENTE   -> ensina invariancia
#     negativo = classe diferente, ROTACAO PROXIMA -> tira o atalho da rotacao

# Trios da RODADA 5: ancora e positivo sao duas janelas normais da MESMA
# rotacao; o negativo e uma janela de desbalanceamento da mesma rotacao. Nada
# de multiclasse -- e o treino que sustenta a contribuicao de open-set.

NORM = {}
IMB = {}
for r in TRAIN_META.itertuples():
    (NORM if r.fault_type == "normal" else IMB).setdefault(r.cond, []).append(r.relpath)
CONDS = sorted(set(NORM) & set(IMB))
print("rotacoes com normal E desbalanceamento no treino: %d" % len(CONDS))
print("   gravacoes normais %d | de desbalanceamento %d"
      % (sum(len(v) for v in NORM.values()), sum(len(v) for v in IMB.values())))
if not CONDS:
    raise SystemExit("nenhuma rotacao tem as duas classes; verifique a particao")


def _janelas(relpath):
    return TRAIN_W[relpath]


class TripletImbalanceDataset(Dataset):
    """Um trio por item, formado dinamicamente (mesma logica do treino original)."""

    def __init__(self, steps=STEPS_PER_EPOCH):
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        cond = CONDS[random.randrange(len(CONDS))]
        Nw = _janelas(NORM[cond][random.randrange(len(NORM[cond]))])
        Iw = _janelas(IMB[cond][random.randrange(len(IMB[cond]))])
        if len(Nw) < 2 or len(Iw) < 1:
            return self.__getitem__(idx)
        i1, i2 = random.sample(range(len(Nw)), 2)
        a = aug_weak(np.asarray(Nw[i1], np.float32))
        p = aug_weak(np.asarray(Nw[i2], np.float32))
        n = np.asarray(Iw[random.randrange(len(Iw))], np.float32)
        return torch.tensor(a), torch.tensor(p), torch.tensor(n)


# %% ===================================================================== R7
# Treino.

LOG = []
CKPTS = {}
for seed in SEEDS:
    print("\n=== semente %d | %s" % (seed, tempo()), flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = EmbeddingNet1D().to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    criterion = TripletMarginLoss(margin=MARGIN, p=2).to(DEVICE)
    loader = DataLoader(TripletImbalanceDataset(), batch_size=BATCH,
                        shuffle=True, drop_last=True, num_workers=0)

    model.train()
    for epoch in range(EPOCHS):
        total, nb = 0.0, 0
        for a, p, n in loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            loss = criterion(model(a), model(p), model(n))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        media = total / max(nb, 1)
        LOG.append(dict(seed=seed, epoch=epoch + 1, loss=media, minutos=round((time.time() - T_START) / 60, 1)))
        print("   epoca %02d/%02d | perda %.4f | %s" % (epoch + 1, EPOCHS, media, tempo()), flush=True)

    path = os.path.join(ROOT, "embedding_model_r7_seed%d.pth" % seed)
    torch.save(model.state_dict(), path)
    CKPTS[seed] = dict(path=path, sha=sha_arquivo(path))
    print("   checkpoint %s  sha %s" % (os.path.basename(path), CKPTS[seed]["sha"]))

LOGDF = pd.DataFrame(LOG)
LOGDF.to_csv(os.path.join(ROOT, "treino_log.csv"), index=False)

# curva da perda por epoca
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
for sd, g in LOGDF.groupby("seed"):
    ax.plot(g.epoch, g.loss, marker="o", ms=3.5, lw=1.4, label="semente %d" % sd)
ax.set_xlabel("epoca")
ax.set_ylabel("perda (triplet, margem %.1f)" % MARGIN)
ax.set_title("Rodada 7 -- convergencia do treino")
ax.grid(alpha=0.3, lw=0.5)
ax.set_xticks(list(LOGDF.epoch.unique()))
if LOGDF.seed.nunique() > 1:
    ax.legend(frameon=False, fontsize=8)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "curva_perda.png"))
try:
    from IPython.display import display
    display(fig)
except Exception:
    pass
plt.close(fig)
print(LOGDF.pivot_table(index="epoch", columns="seed", values="loss").round(4).to_string())
SEED_PRINCIPAL = SEEDS[0]
model = EmbeddingNet1D().to(DEVICE)
model.load_state_dict(torch.load(CKPTS[SEED_PRINCIPAL]["path"], map_location=DEVICE))
model.eval()
print("\navaliando com a semente %d (sha %s) | %s"
      % (SEED_PRINCIPAL, CKPTS[SEED_PRINCIPAL]["sha"], tempo()))


# %% ===================================================================== R8
# Avaliacao no MAFAULDA e embeddings por gravacao, NUMA PASSAGEM SO. Cada
# gravacao de normal, cal e teste e lida uma vez: dela saem os embeddings que
# alimentam o enrolment, a avaliacao e os .npz rotulados.
#
# Enrolment por rotacao, no bloco contiguo de calibracao da gravacao normal
# daquela rotacao. Protocolo do artigo: mu, Sigma + lambda I, T = Q(1-q) dos
# escores da referencia, e a gravacao e decidida quando a porcentagem de
# janelas anomalas passa de 50 %.

def enrol(E_fit, lam=LAMBDA, q=QUANT):
    mu = E_fit.mean(axis=0)
    invC = np.linalg.inv(np.cov(E_fit, rowvar=False) + np.eye(E_fit.shape[1]) * lam)
    d = np.sqrt(np.sum((E_fit - mu) @ invC * (E_fit - mu), axis=1))
    return mu, invC, float(np.quantile(d, q))


def pct_anom(E, mu, invC, thr):
    d = np.sqrt(np.sum((E - mu) @ invC * (E - mu), axis=1))
    return float((d > thr).mean() * 100.0), float(d.mean()), float(d.max())


def guarda(acervo, r, E, nrm):
    acervo.setdefault(r.split, []).append(
        (r.relpath, r.label, r.fault_type, str(r.severity), float(r.cond),
         E.astype(EMB_DTYPE), nrm.astype(EMB_DTYPE)))


ACERVO = {}

# --- 1. gravacoes normais: referencia por rotacao e bloco de teste ----------
print("enrolment por rotacao...", flush=True)
MODELO_COND, NORM_TESTE = {}, {}
for r in meta[meta.fault_type == "normal"].itertuples():
    Xw = windows_norm(r.filepath)
    if len(Xw) < 10:
        continue
    _, i_cal, i_te = blocos_contiguos(len(Xw))
    E, nrm = embed_windows_array(Xw, model, with_norm=True)
    MODELO_COND[r.cond] = enrol(E[i_cal])
    NORM_TESTE[r.cond] = (r.relpath, E[i_te])
    guarda(ACERVO, r, E, nrm)
print("   %d rotacoes com referencia | %s" % (len(MODELO_COND), tempo()))

if not MODELO_COND:
    raise SystemExit("nenhuma gravacao normal produziu referencia; verifique o download")


def rotacao_proxima(cond):
    ks = np.array(sorted(MODELO_COND))
    return float(ks[int(np.argmin(np.abs(ks - cond)))])


# --- 2. especificidade, no bloco de teste das normais -----------------------
linhas = []
for cond, (rel, E) in NORM_TESTE.items():
    mu, invC, thr = MODELO_COND[cond]
    p, sm, sx = pct_anom(E, mu, invC, thr)
    linhas.append(dict(relpath=rel, label="normal", fault_type="normal",
                       severity="normal", cond=cond, split="normal-contiguo",
                       n_win=len(E), pct=p, score_mean=sm, score_max=sx, thr=thr))

# --- 3. sensibilidade, nas gravacoes de falha retidas (e guarda cal tambem) --
alvo = meta[meta.split.isin(("test", "cal"))]
print("avaliando %d gravacoes de falha (test + cal)..." % len(alvo), flush=True)
for i, r in enumerate(alvo.itertuples()):
    Xw = windows_norm(r.filepath)
    if not len(Xw):
        continue
    E, nrm = embed_windows_array(Xw, model, with_norm=True)
    guarda(ACERVO, r, E, nrm)
    c = r.cond if r.cond in MODELO_COND else rotacao_proxima(r.cond)
    mu, invC, thr = MODELO_COND[c]
    p, sm, sx = pct_anom(E, mu, invC, thr)
    linhas.append(dict(relpath=r.relpath, label=r.label, fault_type=r.fault_type,
                       severity=r.severity, cond=r.cond, split=r.split,
                       n_win=len(E), pct=p, score_mean=sm, score_max=sx, thr=thr))
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(alvo), tempo()), flush=True)

AVAL = pd.DataFrame(linhas)
AVAL.to_csv(os.path.join(ROOT, "mafaulda_avaliacao.csv"), index=False)

# --- 4. metricas: so o que e retido conta (normal-contiguo e test) -----------
RET = AVAL[AVAL.split != "cal"]
y = (RET.fault_type != "normal").values
yh = (RET.pct >= DECISION_LEVEL).values
tp, tn = int((y & yh).sum()), int((~y & ~yh).sum())
fp, fn = int((~y & yh).sum()), int((y & ~yh).sum())
MET = dict(n=len(RET), normais=int((~y).sum()), falhas=int(y.sum()),
           acuracia=100.0 * (tp + tn) / max(len(RET), 1),
           sensibilidade=100.0 * tp / max(tp + fn, 1),
           especificidade=100.0 * tn / max(tn + fp, 1),
           nivel=DECISION_LEVEL, tp=tp, tn=tn, fp=fp, fn=fn)
MET["gmean"] = float(np.sqrt(MET["sensibilidade"] * MET["especificidade"]))
por_familia = (RET[RET.fault_type != "normal"]
               .assign(detectada=lambda d: d.pct >= DECISION_LEVEL)
               .groupby("fault_type")
               .agg(gravacoes=("pct", "size"), pct_medio=("pct", "mean"),
                    detectadas=("detectada", "sum")))
por_familia["taxa"] = 100.0 * por_familia.detectadas / por_familia.gravacoes
MET["por_familia"] = json.loads(por_familia.to_json(orient="index"))

# recorte OPEN-SET: as familias que ficaram fora do treino
op = RET[~RET.fault_type.isin(FAM_TREINO) & (RET.fault_type != "normal")]
if len(op):
    MET["open_set"] = dict(
        familias=sorted(op.fault_type.unique()), gravacoes=int(len(op)),
        detectadas=int((op.pct >= DECISION_LEVEL).sum()),
        taxa=float(100.0 * (op.pct >= DECISION_LEVEL).mean()),
        pct_medio=float(op.pct.mean()))
    print("\nOPEN-SET (familias fora do treino: %s): %d de %d detectadas (%.2f %%), "
          "pct medio %.2f" % (", ".join(MET["open_set"]["familias"]),
                              MET["open_set"]["detectadas"], MET["open_set"]["gravacoes"],
                              MET["open_set"]["taxa"], MET["open_set"]["pct_medio"]))
json.dump(MET, open(os.path.join(ROOT, "mafaulda_metricas.json"), "w"), indent=2)
print("\nacuracia %.2f | sensibilidade %.2f | especificidade %.2f | G-mean %.2f"
      % (MET["acuracia"], MET["sensibilidade"], MET["especificidade"], MET["gmean"]))
print(por_familia.to_string())
print("\nLembrete: a particao agora e por gravacao, entao uma queda em relacao aos "
      "96,44 / 96,06 / 100,00 da rodada 5 e esperada e honesta -- nao e regressao.")

# --- 5. embeddings rotulados, um .npz por particao --------------------------
# Faltaram na rodada 5 e travaram o teste da direcao de falha vinda do treino.
for split_alvo, itens in ACERVO.items():
    info = np.array([it[:5] for it in itens], dtype=object)
    embs = [it[5] for it in itens]
    nrms = [it[6] for it in itens]
    offs = np.cumsum([0] + [len(E) for E in embs]).astype(np.int64)
    np.savez_compressed(
        os.path.join(EMB_DIR, "mafaulda_%s_emb.npz" % split_alvo.replace("-", "_")),
        E=np.concatenate(embs), offsets=offs, z_norm=np.concatenate(nrms),
        relpath=info[:, 0].astype(str), label=info[:, 1].astype(str),
        fault_type=info[:, 2].astype(str), severity=info[:, 3].astype(str),
        cond=info[:, 4].astype(float),
        ckpt=CKPTS[SEED_PRINCIPAL]["sha"], window=WINDOW_SIZE, stride=STRIDE,
        scheme="6real_win")
    print("   %-18s %4d gravacoes, %7d janelas" % (split_alvo, len(info), offs[-1]))
del ACERVO


# %% ==================================================================== R9
# Manifesto, zip e download.

manifest = dict(
    rodada=7, gerado_em=time.strftime("%Y-%m-%d %H:%M:%S"),
    checkpoints={str(k): v for k, v in CKPTS.items()},
    checkpoint_principal=CKPTS[SEED_PRINCIPAL]["sha"],
    treino=dict(margin=MARGIN, lr=LR, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                batch=BATCH, seeds=list(SEEDS), max_win_train=MAX_WIN_TRAIN,
                otimizador="Adam", perda="TripletMarginLoss(p=2)"),
    arquitetura=dict(convs="6->32->64->128, kernel 9, padding 4",
                     pooling="MaxPool1d(k=%d, s=%d, p=%d) apos cada ReLU" % (POOL_K, POOL_S, POOL_P),
                     comprimentos="1024 -> 512 -> 256 -> 128",
                     campo_receptivo=campo_receptivo()[0],
                     campo_receptivo_por_camada={n: r for n, r, _ in campo_receptivo()[1]}),
    trios=dict(estrategia="rodada 5, binaria",
               familias_no_treino=list(FAM_TREINO),
               ancora_e_positivo="duas janelas normais da mesma rotacao",
               negativo="janela de desbalanceamento da mesma rotacao",
               aug="aug_weak(shift 64, amp 0.85-1.15, ruido 0.02), apos o z-score"),
    particao=dict(falhas="por gravacao, estratificada por classe, 60/20/20, semente %d" % SEED_SPLIT,
                  normais="contigua no tempo dentro da unica gravacao de cada rotacao, 60/20/20"),
    janela=dict(window=WINDOW_SIZE, stride=STRIDE, norm="z-score por janela e canal"),
    deteccao=dict(lam=LAMBDA, quantil=QUANT, nivel_decisao=DECISION_LEVEL),
    classes_no_treino=int(TRAIN_META.label.nunique()), gravacoes=int(len(meta)),
    rotacoes=int(meta.cond.nunique()),
    metricas=MET, canais=MAF_CH, fs=MAF_FS,
    minutos=round((time.time() - T_START) / 60, 1),
    files=sorted(os.path.relpath(os.path.join(dp, f), ROOT)
                 for dp, _, fs in os.walk(ROOT) for f in fs),
)
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2)[:2000])

ZIP = "/content/rodada7_treino.zip"
os.system("cd %s && zip -q -r -0 %s ." % (ROOT, ZIP))
print("zip %.1f MB" % (os.path.getsize(ZIP) / 1e6))

DRIVE = "/content/drive/MyDrive"
if os.path.isdir(DRIVE):
    shutil.copy(ZIP, os.path.join(DRIVE, os.path.basename(ZIP)))
    for v in CKPTS.values():                     # o checkpoint, solto, para o extrator
        shutil.copy(v["path"], os.path.join(DRIVE, os.path.basename(v["path"])))
    print("copiado para o Drive:", DRIVE)

try:
    from google.colab import files as colab_files
    colab_files.download(ZIP)
except Exception as e:
    print("download automatico indisponivel (%s); pegue pelo Drive." % e)
print("fim |", tempo())
