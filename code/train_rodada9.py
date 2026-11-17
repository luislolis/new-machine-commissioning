#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 9 -- um so modelo, trios por PONTO DE OPERACAO, todo o MAFAULDA usado.

Este .py e a FONTE. O que se abre no Colab e train_rodada9.ipynb, gerado por
`python make_notebook6.py train_rodada9`.

POR QUE
-------
Na rodada 7 (medido em 2026-11-03):
  - os trios usavam so as 33 rotacoes em que normal e desbalanceamento tem o
    MESMO numero no nome do arquivo; o nome e a rotacao medida, numa grade de
    0,2048 Hz, e quase nunca coincide entre pastas;
  - so 415 das 1902 gravacoes de falha eram avaliadas: as 930 gravacoes
    open-set da particao de "treino" nunca entram no treino nem no teste;
  - o modelo saudavel do MAFAULDA era de covariancia cheia, com mu, Sigma e T
    no mesmo bloco, e marcava 11,5 % das janelas normais de teste.

O QUE MUDA
----------
1. PONTO DE OPERACAO. Cada gravacao pertence a um dos 50 pontos da varredura
   normal (code/analise_rotacoes_mafaulda.py -> gravacoes.csv): a k-esima
   rotacao de cada pasta vai para o k-esimo ponto varrido. O ponto 39
   (50,69 Hz) nao tem normal.
2. TRIOS POR PONTO, todos os pontos, um so treino: ancora e positivo sao duas
   janelas normais do ponto p; o negativo e uma janela de desbalanceamento do
   MESMO ponto p. Sao 49 pontos em vez de 33.
3. NORMAIS EM TRES BLOCOS contiguos, com uma janela de guarda entre eles:
       treino     janelas 0..96     (~1 s)    so a rede ve
       enrolment  janelas 98..390   (3 s)     piso de sinal do metodo
       teste      janelas 392..486  (~0,9 s)  especificidade
4. DESBALANCEAMENTO INTEIRO NO TREINO (326 gravacoes; as 7 do ponto 39 nao
   formam trio). As quatro familias open-set INTEIRAS sao teste (1569).
5. EMBEDDINGS DE TODAS AS 1951 GRAVACOES, com o checkpoint novo E com o da
   rodada 7. A decisao (variancia por coordenada, T no enrolment) e feita
   localmente por code/analise_final_v6.py; com o da rodada 7, separa o efeito
   do protocolo de avaliacao do efeito do treino novo.
6. TACOMETRO: a rotacao estimada pelo canal 0 com 50 e com 51,2 kHz, contra o
   numero do nome do arquivo. Decide a taxa de amostragem.

O QUE NAO MUDA
--------------
A receita da rodada 7 inteira: rede com max pooling, janela 1024, passo 512,
z-score por janela, aug_weak, TripletMarginLoss(margin=0.8), Adam 3e-4,
12 epocas, 2000 trios por epoca, lote 64, MAX_WIN_TRAIN 48, semente 42. As
secoes R1 a R3 e as definicoes de R5 sao executadas a partir do proprio
train_rodada7, sem copia.

ANTES DE RODAR
--------------
Na pasta do Drive em que estao train_rodada7 e rerun_extract7 (PASTA_RODADA7,
abaixo), deixe tambem:
    gravacoes.csv                       de code/mafaulda_rotacoes/
O checkpoint embedding_model_r7_seed42.pth e procurado no Drive (a rodada 7 o
deixou na raiz do MyDrive).

O QUE SALVA
-----------
  rodada9_treino.zip
    embedding_model_r9_seed42.pth   checkpoint, com o sha do arquivo
    treino_log.csv, curva_perda.png
    particao.json                   ponto, papel e blocos de cada gravacao
    mafaulda_emb_r9.npz             todas as janelas das 1951 gravacoes
    mafaulda_emb_r7.npz             idem, checkpoint da rodada 7
    tacometro.csv                   rotacao pelo tacometro, gravacao a gravacao
    resumo.json, manifest.json

DEPOIS
------
rerun_extract9.ipynb (IMS, PRONOSTIA e JNU com o checkpoint novo). Localmente:
descompactar em code/rodada9/ e code/rerun9/ e rodar analise_final_v6.py.

TEMPO
-----
GPU T4: ~20 min de leitura para o treino (382 gravacoes), ~15 min de treino,
~50 min para ler e embeddar as 1951 gravacoes com os dois checkpoints.
O zip fica com ~0,5 GB.
"""

# %% ===================================================================== T0
# Drive, GPU e localizacao dos arquivos. Rode esta celula primeiro.

import os
import sys
import re
import json
import time
import subprocess

TESTE = os.environ.get("RODADA9_TESTE") == "1"      # ensaio local, sem Colab

try:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    print("Drive montado em /content/drive")
except Exception as e:
    print("sem Colab (%s); seguindo com a pasta local" % e)

try:
    import torch as _t
    print("GPU: %s" % (_t.cuda.get_device_name(0) if _t.cuda.is_available()
                       else "NAO DISPONIVEL -- Ambiente de execucao > Alterar tipo > T4"))
except Exception:
    pass

# Pasta onde estao train_rodada7, rerun_extract7 e gravacoes.csv. Preenchida,
# a busca nem acontece; deixe "" para procurar no Drive.
PASTA_RODADA7 = "/content/drive/MyDrive/UFPR/orientacoes/mestrado/misael/code"
if TESTE:
    PASTA_RODADA7 = os.environ["PASTA_CODE"]

_PODAR = {".git", ".config", ".ipynb_checkpoints", "__pycache__", "node_modules",
          "sample_data", "IMS_DATA", "JNU-Bearing-Dataset", "rodada6", "rodada7",
          "rodada8", "rodada9", "rerun5", "rerun6", "rerun7", "rerun9", "mafaulda_emb",
          "ieee-phm-2012-data-challenge-dataset", ".shortcut-targets-by-id",
          ".Trash", "Colab Notebooks Backup"}


def _raizes():
    r = [os.getcwd(), "/content"]
    for d in ("/content/drive/MyDrive", "/content/drive/Shareddrives",
              "/content/drive/Shared drives"):
        if os.path.isdir(d):
            r.append(d)
    return [x for i, x in enumerate(r) if x not in r[:i]]


def acha(nome, segundos=60, fundo=6):
    """Procura `nome` com prazo e poda; None se nao achar a tempo."""
    if PASTA_RODADA7:
        p = os.path.join(PASTA_RODADA7, nome)
        if os.path.exists(p):
            return p
    if os.path.exists(nome):
        return os.path.abspath(nome)
    t0 = time.time()
    for raiz in _raizes():
        base_n = raiz.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(raiz):
            if time.time() - t0 > segundos:
                print("   busca parada no prazo de %ds" % segundos)
                return None
            if dirpath.rstrip("/").count("/") - base_n >= fundo:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in _PODAR and not d.startswith(".")]
            if nome in filenames:
                return os.path.join(dirpath, nome)
    return None


def secao_rodada7(base, ini, fim):
    """Codigo de `base` entre as secoes `ini` (None = desde o inicio) e `fim`,
    lido do .py ou do .ipynb. O .py marca as secoes com `# %% ==== R5`; o
    notebook gerado por make_notebook6 com uma celula que comeca em `# R5`."""
    p = acha(base + ".py")
    if p:
        src = open(p, encoding="utf-8").read()
        marca = lambda s: re.search(r"^# %% =+ " + s + r"\s*$", src, re.M)
        a = marca(ini).end() if ini else 0
        b = marca(fim).start()
        print("   %s [%s..%s) <- %s" % (base, ini or "inicio", fim, p))
        return src[a:b]
    p = acha(base + ".ipynb")
    if p:
        nb = json.load(open(p, encoding="utf-8"))
        ped, dentro = [], ini is None
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            if re.match(r"\s*#\s*%s\b" % fim, s):
                break
            if not dentro and re.match(r"\s*#\s*%s\b" % ini, s):
                dentro = True
            if dentro:
                ped.append(s)
        print("   %s [%s..%s) <- %s (%d celulas)" % (base, ini or "inicio", fim, p, len(ped)))
        return "\n\n".join(ped)
    raise SystemExit("nao achei %s.py nem %s.ipynb; ponha a pasta em PASTA_RODADA7" % (base, base))


def sem_magias(src):
    """Comenta as linhas `!comando` do IPython, que nao compilam em exec."""
    return "\n".join(("# " + l) if l.lstrip().startswith("!") else l for l in src.split("\n"))


def troca(src, a, b):
    assert src.count(a) == 1, "trecho esperado uma vez no codigo da rodada 7: %r" % a
    return src.replace(a, b)


ARQ = {n: acha(n) for n in ("gravacoes.csv", "embedding_model_r7_seed42.pth")}
for n, p in ARQ.items():
    print("   %-32s %s" % (n, p or "NAO ENCONTRADO"))
if ARQ["gravacoes.csv"] is None:
    raise SystemExit("falta gravacoes.csv: copie code/mafaulda_rotacoes/gravacoes.csv "
                     "para %s" % PASTA_RODADA7)
if ARQ["embedding_model_r7_seed42.pth"] is None:
    print("   (sem o checkpoint da rodada 7, a celula T6 so embedda com o novo)")


# %% ===================================================================== T1
# Preparo: R1-R3 da rodada 7 (constantes, rede, metadados do MAFAULDA) e as
# definicoes de R5 (leitura, janelas, aug_weak), executados do proprio codigo.

if not TESTE:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "kagglehub"], check=False)

_pre = sem_magias(secao_rodada7("train_rodada7", None, "R4"))
_pre = troca(_pre, 'ROOT = "/content/rodada7"',
             'ROOT = %r' % (os.environ["RODADA9_ROOT"] if TESTE else "/content/rodada9"))
if TESTE:
    _pre = troca(_pre, "import kagglehub", "kagglehub = None")
    _pre = troca(_pre, 'MAF_PATH = kagglehub.dataset_download("vuxuancu/mafaulda-full")',
                 'MAF_PATH = os.environ["MAF_TESTE"]')
exec(compile(_pre, "train_rodada7[R1-R3]", "exec"), globals())

FRAC = (0.60, 0.20, 0.20)        # so para a assinatura de blocos_contiguos em R5
SEED_SPLIT = 42
_r5 = secao_rodada7("train_rodada7", "R5", "R6")
assert 'print("passagem unica' in _r5
_r5 = _r5.split('print("passagem unica')[0]
exec(compile(_r5, "train_rodada7[R5, definicoes]", "exec"), globals())

if TESTE:
    EPOCHS, STEPS_PER_EPOCH = 1, 128
print("preparo concluido | %s" % tempo(), flush=True)


# %% ===================================================================== T2
# Ponto de operacao e papel de cada gravacao.

B_TREINO, B_ENROL, B_TESTE = (0, 97), (98, 391), (392, 487)
N_JAN = B_TESTE[1]                                   # 487 janelas por gravacao
# uma janela de guarda: a janela i cobre [512 i, 512 i + 1024), entao i e i + 2
# nao compartilham amostras
assert B_ENROL[0] >= B_TREINO[1] + 1 and B_TESTE[0] >= B_ENROL[1] + 1
assert B_ENROL[1] - B_ENROL[0] == 293                # 3 s de sinal

GRAV = pd.read_csv(ARQ["gravacoes.csv"])
meta = meta.merge(GRAV[["relpath", "ponto", "f", "f_nom", "tem_normal", "caso", "marcas"]],
                  on="relpath", how="left", validate="one_to_one")
assert meta.ponto.notna().all(), "gravacoes sem ponto: %s" % meta[meta.ponto.isna()].relpath.tolist()[:5]
assert np.allclose(meta.cond.values, meta.f.values), "rotacao do nome difere da do gravacoes.csv"
if not TESTE:
    assert len(meta) == 1951 and len(GRAV) == 1951, "esperadas 1951 gravacoes"
meta["ponto"] = meta.ponto.astype(int)

FAM_OPEN = ("horizontal-misalignment", "vertical-misalignment",
            "underhang-bearing", "overhang-bearing")
PONTOS = sorted(set(meta[meta.fault_type == "normal"].ponto)
                & set(meta[meta.fault_type == "imbalance"].ponto))
papel = np.where(meta.fault_type == "normal", "normal: treino + enrolment + teste",
        np.where(meta.fault_type == "imbalance",
                 np.where(meta.ponto.isin(PONTOS), "treino", "fora (ponto sem normal)"),
                 "teste open-set"))
meta["papel"] = papel
assert not ((meta.papel == "treino") & meta.fault_type.isin(FAM_OPEN)).any()
print("pontos com trio: %d" % len(PONTOS))
print(meta.groupby(["fault_type", "papel"]).size().to_string())
json.dump(dict(blocos_normal=dict(treino=B_TREINO, enrolment=B_ENROL, teste=B_TESTE),
               pontos_com_trio=[int(p) for p in PONTOS],
               gravacoes={r.relpath: dict(ponto=int(r.ponto), papel=r.papel)
                          for r in meta.itertuples()}),
          open(os.path.join(ROOT, "particao.json"), "w"), indent=1)


# %% ===================================================================== T3
# Janelas de treino: das normais, so o bloco de treino; do desbalanceamento,
# a gravacao inteira. Mesmo sorteio de MAX_WIN_TRAIN janelas da rodada 7.

rng_w = np.random.default_rng(SEED_SPLIT)
TRAIN_W = {}
alvo = meta[(meta.fault_type == "normal") | (meta.papel == "treino")]
print("lendo %d gravacoes para o treino..." % len(alvo), flush=True)
for i, r in enumerate(alvo.itertuples()):
    Xw = windows_norm(r.filepath)
    if r.fault_type == "normal":
        assert len(Xw) == N_JAN, "%s tem %d janelas" % (r.relpath, len(Xw))
        pool = np.arange(*B_TREINO)
    else:
        pool = np.arange(len(Xw))
    sel = pool if len(pool) <= MAX_WIN_TRAIN else rng_w.choice(pool, MAX_WIN_TRAIN, replace=False)
    TRAIN_W[r.relpath] = Xw[np.sort(sel)].astype(np.float16)
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(alvo), tempo()), flush=True)
print("cache de treino: %d gravacoes, %d janelas | %s"
      % (len(TRAIN_W), sum(len(v) for v in TRAIN_W.values()), tempo()))


# %% ===================================================================== T4
# Trios por ponto de operacao. Mesma logica do R6 da rodada 7; so a chave muda,
# de rotacao exata para ponto.

NORM, IMB = {}, {}
for r in alvo.itertuples():
    (NORM if r.fault_type == "normal" else IMB).setdefault(int(r.ponto), []).append(r.relpath)
CONDS = [p for p in PONTOS if p in NORM and p in IMB]
assert len(CONDS) == len(PONTOS)
print("pontos nos trios: %d | gravacoes normais %d | de desbalanceamento %d"
      % (len(CONDS), sum(len(NORM[p]) for p in CONDS), sum(len(IMB[p]) for p in CONDS)))


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


# %% ===================================================================== T5
# Treino. Identico ao R7 da rodada 7.

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
        LOG.append(dict(seed=seed, epoch=epoch + 1, loss=media,
                        minutos=round((time.time() - T_START) / 60, 1)))
        print("   epoca %02d/%02d | perda %.4f | %s" % (epoch + 1, EPOCHS, media, tempo()), flush=True)

    path = os.path.join(ROOT, "embedding_model_r9_seed%d.pth" % seed)
    torch.save(model.state_dict(), path)
    CKPTS[seed] = dict(path=path, sha=sha_arquivo(path))
    print("   checkpoint %s  sha %s" % (os.path.basename(path), CKPTS[seed]["sha"]))

LOGDF = pd.DataFrame(LOG)
LOGDF.to_csv(os.path.join(ROOT, "treino_log.csv"), index=False)
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
for sd, g in LOGDF.groupby("seed"):
    ax.plot(g.epoch, g.loss, marker="o", ms=3.5, lw=1.4, label="semente %d" % sd)
ax.set_xlabel("epoca")
ax.set_ylabel("perda (triplet, margem %.1f)" % MARGIN)
ax.set_title("Rodada 9 -- convergencia do treino")
ax.grid(alpha=0.3, lw=0.5)
ax.set_xticks(list(LOGDF.epoch.unique()))
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
SEED_PRINCIPAL = SEEDS[0]
del TRAIN_W


# %% ===================================================================== T6
# Embeddings de TODAS as gravacoes, com o checkpoint novo e o da rodada 7, e a
# rotacao pelo tacometro. Cada gravacao e lida uma vez so.

def carrega_modelo(path):
    net = EmbeddingNet1D().to(DEVICE)
    try:
        st = torch.load(path, map_location=DEVICE, weights_only=True)
    except Exception:
        st = torch.load(path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(st, strict=True)
    net.eval()
    return net


MODELOS = {"r9": carrega_modelo(CKPTS[SEED_PRINCIPAL]["path"])}
SHAS = {"r9": CKPTS[SEED_PRINCIPAL]["sha"]}
if ARQ["embedding_model_r7_seed42.pth"]:
    MODELOS["r7"] = carrega_modelo(ARQ["embedding_model_r7_seed42.pth"])
    SHAS["r7"] = sha_arquivo(ARQ["embedding_model_r7_seed42.pth"])
print("checkpoints: %s" % SHAS)


def janelas_de(X):
    """O corpo de windows_norm, a partir do sinal ja lido."""
    wins = [X[s:s + WINDOW_SIZE, :].T
            for s in range(0, X.shape[0] - WINDOW_SIZE + 1, STRIDE)]
    if not wins:
        return np.empty((0, 6, WINDOW_SIZE), np.float32)
    Xw = np.stack(wins, axis=0)
    mean = Xw.mean(axis=2, keepdims=True)
    std = Xw.std(axis=2, keepdims=True) + 1e-8
    return (Xw - mean) / std


def tacometro(x, nome_hz):
    """Pulsos do canal 0 e pico do espectro, nas duas taxas candidatas."""
    x = np.asarray(x, np.float64)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    thr = 0.5 * (lo + hi)
    up = np.flatnonzero((x[:-1] < thr) & (x[1:] >= thr))
    per = (up[-1] - up[0]) / (len(up) - 1) if len(up) >= 3 else np.nan
    N = len(x)
    P = np.abs(np.fft.rfft(x - x.mean()))
    k = np.arange(len(P))
    faixa = (k * 51200.0 / N >= 5.0) & (k * 51200.0 / N <= 70.0)
    kp = int(k[faixa][np.argmax(P[faixa])])
    return dict(n_amostras=N, pulsos=int(len(up)), amostras_por_pulso=per,
                f_pulsos_51200=51200.0 / per, f_pulsos_50000=50000.0 / per,
                bin_pico=kp, f_pico_51200=kp * 51200.0 / N, f_pico_50000=kp * 50000.0 / N,
                f_nome=nome_hz, bin_nome=nome_hz / (51200.0 / N))


_r0 = meta.iloc[0]
_X0 = read_acc_only(_r0.filepath).to_numpy(dtype=np.float32)
assert np.array_equal(janelas_de(_X0), windows_norm(_r0.filepath)), "janelas_de difere de windows_norm"

ordem = meta.sort_values("relpath").reset_index(drop=True)
ACC = {t: [] for t in MODELOS}
TACO = []
print("embeddando %d gravacoes com %s..." % (len(ordem), ", ".join(MODELOS)), flush=True)
for i, r in enumerate(ordem.itertuples()):
    df = pd.read_csv(r.filepath, header=None)
    if df.shape[1] < 8:
        raise ValueError("%s tem %d colunas" % (r.relpath, df.shape[1]))
    Xw = janelas_de(df.iloc[:, 1:7].to_numpy(dtype=np.float32))
    for t, net in MODELOS.items():
        ACC[t].append(embed_windows_array(Xw, net).astype(EMB_DTYPE))
    TACO.append(dict(relpath=r.relpath, fault_type=r.fault_type, ponto=int(r.ponto),
                     **tacometro(df.iloc[:, 0].to_numpy(), float(r.cond))))
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(ordem), tempo()), flush=True)

TACODF = pd.DataFrame(TACO)
TACODF.to_csv(os.path.join(ROOT, "tacometro.csv"), index=False)
for t, lst in ACC.items():
    offs = np.cumsum([0] + [len(E) for E in lst]).astype(np.int64)
    np.savez_compressed(
        os.path.join(ROOT, "mafaulda_emb_%s.npz" % t),
        E=np.concatenate(lst), offsets=offs,
        relpath=ordem.relpath.values.astype(str), label=ordem.label.values.astype(str),
        fault_type=ordem.fault_type.values.astype(str),
        severity=ordem.severity.values.astype(str), cond=ordem.cond.values.astype(float),
        ponto=ordem.ponto.values.astype(int), ckpt=SHAS[t],
        window=WINDOW_SIZE, stride=STRIDE, scheme="6real_win")
    print("   mafaulda_emb_%s.npz: %d gravacoes, %d janelas" % (t, len(lst), offs[-1]))
del ACC

# taxa de amostragem: o numero do nome e o pico do tacometro em qual taxa?
d512 = np.abs(TACODF.f_pico_51200 - TACODF.f_nome)
d500 = np.abs(TACODF.f_pico_50000 - TACODF.f_nome)
print("\ntacometro: |pico - nome| mediano %.4f Hz supondo 51,2 kHz e %.4f Hz supondo 50 kHz"
      % (d512.median(), d500.median()))
print("razao nome / pulsos a 51,2 kHz: mediana %.4f"
      % (TACODF.f_nome / TACODF.f_pulsos_51200).median())


# %% ===================================================================== T7
# Resumo rapido do protocolo novo. O numero que vale e o de analise_final_v6.py.

def _resumo(t):
    z = np.load(os.path.join(ROOT, "mafaulda_emb_%s.npz" % t))
    E, o = z["E"].astype(np.float64), z["offsets"]
    pt, ft = z["ponto"], z["fault_type"]
    mod = {}
    esp = []
    for i in np.flatnonzero(ft == "normal"):
        X = E[o[i]:o[i + 1]]
        Z = X[B_ENROL[0]:B_ENROL[1]]
        mu, var = Z.mean(0), Z.var(0, ddof=1) + 1e-12
        s = lambda Y: np.sqrt(((Y - mu) ** 2 / var).sum(1))
        T = float(np.quantile(s(Z), QUANT))
        mod[int(pt[i])] = (mu, var, T)
        esp.append(100.0 * np.mean(s(X[B_TESTE[0]:B_TESTE[1]]) > T))
    det = []
    for i in np.flatnonzero(np.isin(ft, FAM_OPEN)):
        if int(pt[i]) not in mod:
            continue
        mu, var, T = mod[int(pt[i])]
        X = E[o[i]:o[i + 1]]
        det.append(100.0 * np.mean(np.sqrt(((X - mu) ** 2 / var).sum(1)) > T))
    esp, det = np.array(esp), np.array(det)
    r = dict(normais_saudaveis=int((esp < DECISION_LEVEL).sum()), normais=int(len(esp)),
             janelas_normais_marcadas=float(esp.mean()),
             open_set_detectadas=int((det >= DECISION_LEVEL).sum()), open_set=int(len(det)))
    print("%s: normais %d/%d saudaveis, %.2f %% das janelas normais marcadas | open-set %d/%d"
          % (t, r["normais_saudaveis"], r["normais"], r["janelas_normais_marcadas"],
             r["open_set_detectadas"], r["open_set"]))
    return r


RESUMO = {t: _resumo(t) for t in MODELOS}
json.dump(RESUMO, open(os.path.join(ROOT, "resumo.json"), "w"), indent=2)


# %% ===================================================================== T8
# Manifesto, zip e download.

manifest = dict(
    rodada=9, gerado_em=time.strftime("%Y-%m-%d %H:%M:%S"),
    checkpoints={str(k): v for k, v in CKPTS.items()},
    checkpoint_principal=CKPTS[SEED_PRINCIPAL]["sha"], embeddings=SHAS,
    treino=dict(margin=MARGIN, lr=LR, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                batch=BATCH, seeds=list(SEEDS), max_win_train=MAX_WIN_TRAIN,
                otimizador="Adam", perda="TripletMarginLoss(p=2)"),
    trios=dict(estrategia="por ponto de operacao (gravacoes.csv)", pontos=len(CONDS),
               ancora_e_positivo="duas janelas do bloco de treino da normal do ponto",
               negativo="janela de desbalanceamento do mesmo ponto",
               aug="aug_weak(shift 64, amp 0.85-1.15, ruido 0.02), apos o z-score"),
    blocos_normal=dict(treino=B_TREINO, enrolment=B_ENROL, teste=B_TESTE),
    gravacoes=int(len(meta)), resumo=RESUMO, teste_local=TESTE,
    minutos=round((time.time() - T_START) / 60, 1),
    files=sorted(os.path.relpath(os.path.join(dp, f), ROOT)
                 for dp, _, fs in os.walk(ROOT) for f in fs),
)
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2)[:2000])

if not TESTE:
    import shutil
    ZIP = "/content/rodada9_treino.zip"
    os.system("cd %s && zip -q -r -0 %s ." % (ROOT, ZIP))
    print("zip %.1f MB" % (os.path.getsize(ZIP) / 1e6))
    DRIVE = "/content/drive/MyDrive"
    if os.path.isdir(DRIVE):
        shutil.copy(ZIP, os.path.join(DRIVE, os.path.basename(ZIP)))
        for v in CKPTS.values():             # o checkpoint, solto, para o extrator
            shutil.copy(v["path"], os.path.join(DRIVE, os.path.basename(v["path"])))
        print("copiado para o Drive:", DRIVE)
    try:
        from google.colab import files as colab_files
        colab_files.download(ZIP)
    except Exception as e:
        print("download automatico indisponivel (%s); pegue pelo Drive." % e)
print("fim |", tempo())
