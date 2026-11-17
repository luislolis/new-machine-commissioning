#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 14 -- IMS, PRONOSTIA e JNU com os seis checkpoints do train_rodada14 (plano-rodada14.md).

Este .py e a FONTE (gerado por gera_rodada14.py a partir do rerun_extract10). O que se abre no
Colab e rerun_extract14.ipynb, gerado por `python make_notebook6.py rerun_extract14`.

O QUE FAZ
---------
Cada arquivo e lido UMA vez e embeddado com todos os checkpoints encontrados:
    r14b0 ... r14b2   embedding_model_r14b0.pth ...   (binario, sementes 42, 43, 44)
    r14m0 ... r14m2   embedding_model_r14m0.pth ...   (multiclasse, sementes 42, 43, 44)
    r9                embedding_model_r9_seed42.pth   (se estiver no Drive)
So o esquema que o artigo usa e extraido: repeticao ciclica dos canais e z-score por janela
(rep_win), nas tres bases. Os leitores, o janelamento, a regra de canais e a deteccao dos
arquivos truncados do PRONOSTIA sao os do rerun_extract10, sem mudanca.

O checkpoint r9 serve de FIDELIDADE: localmente, os embeddings dele aqui tem de coincidir com os
de code/rerun9/. Sem o r9 no Drive, a conferencia e pulada.

ORDEM DOS NOTEBOOKS
-------------------
    1. train_rodada14.ipynb    treina os seis e deixa os checkpoints na raiz do Drive
    2. rerun_extract14.ipynb   este
    3. analise local           descompactar os zips em code/rerun14/ e
                               rodar code/analise_r14.py

SAIDA
-----
Uma pasta por checkpoint, com os mesmos nomes de arquivo de code/rerun9/:
    <tag>/A/jnu/jnu_rep_win_emb.npz
    <tag>/A/pronostia/pronostia_rep_win_BearingX_Y_emb.npz
    <tag>/ims_rep_win/ims_<seq>_b<k>_emb.npz
e um zip por checkpoint, rerun14_<tag>.zip (~350 MB cada), copiado para o Drive.

TEMPO
-----
GPU T4, ~1 h: a leitura dos arquivos domina, e os sete checkpoints embeddam cada lote ja carregado.
"""

# %% ===================================================================== E0
# Setup. Nada aqui treina.

import os, re, json, glob, time, hashlib, shutil, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TESTE = os.environ.get("RERUN14_TESTE") == "1"      # ensaio local com dados falsos

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cpu" and not TESTE:
    print("AVISO: runtime em CPU. Ambiente de execucao > Alterar tipo > GPU T4.")

WINDOW_SIZE = 1024
STRIDE = 512
EMB_DTYPE = np.float16
POOL_K, POOL_S, POOL_P = 4, 2, 1
TAGS14 = tuple("r14%s%d" % (o, k) for k in range(3) for o in ("b", "m"))   # train_rodada14

ROOT = os.environ["RERUN14_ROOT"] if TESTE else "/content/rerun14"
DADOS = os.environ["RERUN14_DADOS"] if TESTE else "/content"
os.makedirs(ROOT, exist_ok=True)
T_START = time.time()


def tempo():
    return "%.1f min" % ((time.time() - T_START) / 60)


try:
    from google.colab import drive
    _NO_COLAB = True
except Exception:
    _NO_COLAB = False
if _NO_COLAB and not TESTE:
    if not os.path.isdir("/content/drive/MyDrive"):
        try:
            drive.mount("/content/drive", force_remount=True)
        except Exception as e:
            print("O Drive nao montou (%s). Para resolver:" % e)
            print("  1. numa celula nova: from google.colab import drive; drive.mount('/content/drive', force_remount=True)")
            print("  2. na janela que abrir, escolha a conta do Drive onde estao os arquivos e marque TODAS as permissoes")
            print("  3. se insistir: Ambiente de execucao > Desconectar e excluir ambiente de execucao, e rode de novo")
            raise SystemExit("sem o Drive montado nao da para seguir")
    print("Drive montado em /content/drive")
else:
    print("sem Colab; seguindo com a pasta local")


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


NOMES = {t: "embedding_model_%s.pth" % t for t in TAGS14}
NOMES["r9"] = "embedding_model_r9_seed42.pth"
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
faltam = [t for t in NOMES if t != "r9" and t not in MODELOS]
if faltam:
    raise SystemExit("faltam os checkpoints %s; rode o train_rodada14 antes" % faltam)

with torch.no_grad():
    _t = torch.zeros(1, 6, WINDOW_SIZE).to(DEVICE)
    _comp = [_t.shape[-1]]
    for _c in next(iter(MODELOS.values())).feat:
        _t = _c(_t)
        if isinstance(_c, nn.MaxPool1d):
            _comp.append(_t.shape[-1])
    assert _comp == [1024, 512, 256, 128], "geometria do pooling: %s" % _comp

SUB = {t: {k: os.path.join(ROOT, t, *k.split("/")) for k in ("A/jnu", "A/pronostia", "ims_rep_win")}
       for t in MODELOS}
for t in SUB.values():
    for d in t.values():
        os.makedirs(d, exist_ok=True)


# %% ===================================================================== E2
# Funcoes. Janelamento, normalizacao e regra de canais identicas ao rerun_extract9.

@torch.no_grad()
def embed_todos(Xw, batch=256):
    """(N, 6, L) -> {tag: (N, 128) float16}, um lote carregado para todos os checkpoints."""
    out = {t: [] for t in MODELOS}
    for i in range(0, len(Xw), batch):
        xb = torch.tensor(Xw[i:i + batch]).to(DEVICE)
        for t, net in MODELOS.items():
            out[t].append(net(xb).cpu().numpy())
    return {t: (np.vstack(v) if v else np.empty((0, 128), np.float32)).astype(EMB_DTYPE)
            for t, v in out.items()}


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
        np.savez_compressed(path, E=np.concatenate(embs).astype(EMB_DTYPE), offsets=offsets,
                            ckpt=CKPT[t]["pesos"], window=WINDOW_SIZE, stride=STRIDE,
                            scheme="rep_win", **meta)
    print("   salvo %-40s %8d janelas x %d checkpoints" % (nome, offsets[-1], len(por_tag)))


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

jnu = {t: [] for t in MODELOS}
for p in meta_jnu["path"]:
    for t, E in embed_todos(jnu_rep_win(read_jnu_csv_1ch(p))).items():
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
    embs = {t: [] for t in MODELOS}
    t0s = []
    for f in files:
        df = read_pronostia_df(f)
        t0s.append(clock_seconds(df))
        for t, E in embed_todos(pronostia_rep_win(df)).items():
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
    embs = {(t, b): [] for t in MODELOS for b in IMS_BEARINGS[seq]}
    t0 = time.time()
    for i, f in enumerate(files):
        if i and i % 500 == 0:
            el = time.time() - t0
            print("   %s %5d/%d  %.0f s, ~%.0f s restantes"
                  % (seq, i, len(files), el, el * (len(files) - i) / i), flush=True)
        raw = pd.read_csv(f, sep=r"\s+", header=None).values.astype(np.float32)
        for b, cols in IMS_BEARINGS[seq].items():
            for t, E in embed_todos(ims_rep_win(raw, cols)).items():
                embs[(t, b)].append(E)
    names = np.array([os.path.basename(f) for f in files])
    print("%s: %d snapshots | %s" % (seq, len(files), tempo()))
    for b in IMS_BEARINGS[seq]:
        salva("ims_rep_win", "ims_%s_%s" % (seq, b), {t: embs[(t, b)] for t in MODELOS},
              files=names, channels=np.array(IMS_BEARINGS[seq][b]), n_columns=ncols)
    del embs


for seq in ("2nd_test", "1st_test", "4th_test"):
    run_ims(seq)


# %% ===================================================================== G
# Manifesto, zips, copia para o Drive e download.

manifest = dict(
    rodada=10, checkpoints=CKPT, esquema="rep_win (repeticao ciclica, z-score por janela)",
    window=WINDOW_SIZE, stride=STRIDE, emb_dtype=str(np.dtype(EMB_DTYPE)),
    ims_columns={k: v[1] for k, v in IMS_NCOL.items()}, ims_bearings=IMS_BEARINGS,
    pronostia_locations=LOC, pronostia_conditions=PRON_COND, teste_local=TESTE,
    minutos=round((time.time() - T_START) / 60, 1),
    files={t: sorted(os.path.relpath(os.path.join(dp, f), os.path.join(ROOT, t))
                     for dp, _, fs in os.walk(os.path.join(ROOT, t)) for f in fs) for t in MODELOS},
)
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
for t in MODELOS:
    shutil.copy(os.path.join(ROOT, "manifest.json"), os.path.join(ROOT, t, "manifest.json"))
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2)[:3000])

if not TESTE:
    ZIPS = []
    for t in MODELOS:
        z = "/content/rerun14_%s.zip" % t
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
