#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 10 -- cinco dobras: o treino da rodada 9 repetido com outras janelas e outra semente.

Este .py e a FONTE. O que se abre no Colab e train_rodada10.ipynb, gerado por
`python make_notebook6.py train_rodada10`.

POR QUE
-------
A rodada 9 treinou uma vez, com a semente 42, e as Limitacoes do artigo dizem
que a variacao entre treinos nao foi medida. Aqui o treino da rodada 9 e
repetido cinco vezes, e cada repeticao muda as duas coisas que um treino sorteia:
  - a SEMENTE: pesos iniciais, sorteio dos trios, aug_weak, ordem dos lotes;
  - as JANELAS de dados que a rede ve e as que medem a especificidade.

DOBRAS
------
Cada normal (487 janelas) tem os tres blocos da rodada 9 -- treino 97, guarda 1,
enrolment 293, guarda 1, teste 95 --, e na dobra f o arranjo inteiro e girado de
97 f janelas, dando a volta no fim da gravacao:
    dobra 0  treino   0- 96 | enrolment  98-390        | teste 392-486  (= rodada 9)
    dobra 1  treino  97-193 | enrolment 195-486, 0     | teste   2- 95
    dobra 2  treino 194-290 | enrolment 292-486, 0- 97 | teste  99-192
    dobra 3  treino 291-387 | enrolment 389-486, 0-194 | teste 196-289
    dobra 4  treino 388-484 | enrolment 486, 0-291     | teste 293-386
Toda janela de 0 a 484 e de treino em exatamente uma dobra e os cinco blocos de
teste sao disjuntos: e um 5-fold em blocos sobre o tempo de cada normal. A janela
i cobre as amostras [512 i, 512 i + 1024), entao duas janelas vizinhas
compartilham amostras; uma janela de teste que encostaria numa de treino sai do
teste, e por isso as dobras 1 a 4 testam 94 janelas em vez de 95.

O desbalanceamento continua inteiro no treino, como na rodada 9, mas as 48
janelas sorteadas de cada gravacao mudam com a semente da dobra. As quatro
familias open-set continuam inteiras no teste em todas as dobras.

A dobra 0, com a semente 42, refaz a rodada 9: mesmos blocos e mesmos sorteios.
O treino em GPU nao e reprodutivel bit a bit, entao o checkpoint pode diferir do
da rodada 9; a diferenca fica registrada no manifesto.

O QUE NAO MUDA
--------------
A receita da rodada 9 inteira: rede com max pooling, janela 1024, passo 512,
z-score por janela, aug_weak, TripletMarginLoss(margin=0.8), Adam 3e-4, 12
epocas, 2000 trios por epoca, lote 64, MAX_WIN_TRAIN 48, trios por ponto de
operacao. As secoes R1 a R3 e as definicoes de R5 sao executadas a partir do
proprio train_rodada7, sem copia, como na rodada 9.

ANTES DE RODAR
--------------
Na pasta do Drive em que esta train_rodada7 (PASTA_RODADA7, abaixo), deixe:
    gravacoes.csv                       de code/mafaulda_rotacoes/
O checkpoint embedding_model_r9_seed42.pth (raiz do MyDrive) e procurado para
comparar com a dobra 0; sem ele, a comparacao e pulada.

O QUE SALVA
-----------
  rodada10_treino.zip
    embedding_model_r10_d0.pth ... d4.pth   um checkpoint por dobra
    mafaulda_emb_d0.npz ... d4.npz          todas as janelas das 1951 gravacoes
    particao.json                           blocos de cada dobra, ponto e papel
    treino_log.csv, curva_perda.png, resumo.json, manifest.json
Os cinco checkpoints tambem vao soltos para a raiz do MyDrive, para o extrator.

DEPOIS
------
rerun_extract10.ipynb (IMS, PRONOSTIA e JNU com os cinco checkpoints). Localmente:
descompactar em code/rodada10/ e code/rerun10/ e rodar code/analise_dobras.py.

TEMPO
-----
GPU T4: ~20 min de leitura para o treino, ~15 min de treino por dobra (~75 min)
e ~45 min para ler e embeddar as 1951 gravacoes com os cinco checkpoints, cerca
de 2 h 20 ao todo. O zip fica com ~1,1 GB.
"""

# %% ===================================================================== T0
# Drive, GPU e localizacao dos arquivos. Rode esta celula primeiro.

import os
import sys
import re
import json
import time
import subprocess

TESTE = os.environ.get("RODADA10_TESTE") == "1"     # ensaio local, sem Colab

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

# Pasta onde estao train_rodada7 e gravacoes.csv. Preenchida, a busca nem
# acontece; deixe "" para procurar no Drive.
PASTA_RODADA7 = "/content/drive/MyDrive/UFPR/orientacoes/mestrado/misael/code"
if TESTE:
    PASTA_RODADA7 = os.environ["PASTA_CODE"]

_PODAR = {".git", ".config", ".ipynb_checkpoints", "__pycache__", "node_modules",
          "sample_data", "IMS_DATA", "JNU-Bearing-Dataset", "rodada6", "rodada7",
          "rodada8", "rodada9", "rodada10", "rerun5", "rerun6", "rerun7", "rerun9",
          "rerun10", "mafaulda_emb", "ieee-phm-2012-data-challenge-dataset",
          ".shortcut-targets-by-id", ".Trash", "Colab Notebooks Backup"}


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


ARQ = {n: acha(n) for n in ("gravacoes.csv", "embedding_model_r9_seed42.pth")}
for n, p in ARQ.items():
    print("   %-32s %s" % (n, p or "NAO ENCONTRADO"))
if ARQ["gravacoes.csv"] is None:
    raise SystemExit("falta gravacoes.csv: copie code/mafaulda_rotacoes/gravacoes.csv "
                     "para %s" % PASTA_RODADA7)
if ARQ["embedding_model_r9_seed42.pth"] is None:
    print("   (sem o checkpoint da rodada 9, a dobra 0 nao e comparada com ele)")


# %% ===================================================================== T1
# Preparo: R1-R3 da rodada 7 (constantes, rede, metadados do MAFAULDA) e as
# definicoes de R5 (leitura, janelas, aug_weak), executados do proprio codigo.

if not TESTE:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "kagglehub"], check=False)

_pre = sem_magias(secao_rodada7("train_rodada7", None, "R4"))
_pre = troca(_pre, 'ROOT = "/content/rodada7"',
             'ROOT = %r' % (os.environ["RODADA10_ROOT"] if TESTE else "/content/rodada10"))
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
# Ponto de operacao e papel de cada gravacao (como na rodada 9) e os blocos de
# cada dobra.

N_DOBRAS = 5
SEMENTES = tuple(42 + f for f in range(N_DOBRAS))
B_TREINO, B_ENROL, B_TESTE = (0, 97), (98, 391), (392, 487)     # rodada 9 = dobra 0
N_JAN = B_TESTE[1]                                  # 487 janelas por gravacao
DESLOC = N_JAN // N_DOBRAS                          # 97 janelas por dobra


def blocos(f):
    """Treino, enrolment e teste da dobra f: o arranjo da rodada 9 girado de
    DESLOC * f janelas. Duas janelas a distancia 1 compartilham amostras, entao
    o enrolment nao encosta no treino e o teste nao encosta em nenhum dos dois."""
    gira = lambda a, b: np.sort((DESLOC * f + np.arange(a, b)) % N_JAN)
    tr, en, te = gira(*B_TREINO), gira(*B_ENROL), gira(*B_TESTE)
    toca = lambda A: set((A - 1).tolist()) | set((A + 1).tolist())
    en = np.array([i for i in en if i not in toca(tr)], dtype=np.int64)
    te = np.array([i for i in te if i not in toca(tr) | toca(en)], dtype=np.int64)
    return tr, en, te


BLOCOS = {f: blocos(f) for f in range(N_DOBRAS)}
assert all(np.array_equal(a, np.arange(*b)) for a, b in zip(BLOCOS[0], (B_TREINO, B_ENROL, B_TESTE))), \
    "a dobra 0 tem de ser a rodada 9"
for f, (tr, en, te) in BLOCOS.items():
    assert len(tr) == 97 and len(en) == 293 and len(te) >= 94, (f, len(tr), len(en), len(te))
    for A, B in ((tr, en), (tr, te), (en, te)):
        assert np.abs(A[:, None] - B[None, :]).min() >= 2, "dobra %d: blocos compartilham amostras" % f
_tr = np.concatenate([BLOCOS[f][0] for f in BLOCOS])
_te = np.concatenate([BLOCOS[f][2] for f in BLOCOS])
assert len(set(_tr.tolist())) == len(_tr) and len(set(_te.tolist())) == len(_te), \
    "uma janela treina ou testa em mais de uma dobra"
for f, (tr, en, te) in BLOCOS.items():
    print("dobra %d (semente %d): treino %d-%d | enrolment %d janelas | teste %d janelas (%d-%d)"
          % (f, SEMENTES[f], tr[0], tr[-1], len(en), len(te), te[0], te[-1]))

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
json.dump(dict(dobras={str(f): dict(semente=SEMENTES[f], treino=tr.tolist(), enrolment=en.tolist(),
                                    teste=te.tolist())
                       for f, (tr, en, te) in BLOCOS.items()},
               pontos_com_trio=[int(p) for p in PONTOS],
               gravacoes={r.relpath: dict(ponto=int(r.ponto), papel=r.papel)
                          for r in meta.itertuples()}),
          open(os.path.join(ROOT, "particao.json"), "w"), indent=1)


# %% ===================================================================== T3
# Janelas de treino de cada dobra, com uma leitura so de cada gravacao: das
# normais, o bloco de treino da dobra; do desbalanceamento, a gravacao inteira.
# Cada dobra sorteia as suas MAX_WIN_TRAIN janelas com a propria semente, na
# mesma ordem da rodada 9, de modo que a dobra 0 repete o sorteio dela.

RNG_W = {f: np.random.default_rng(SEMENTES[f]) for f in range(N_DOBRAS)}
TRAIN_W = {f: {} for f in range(N_DOBRAS)}
alvo = meta[(meta.fault_type == "normal") | (meta.papel == "treino")]
print("lendo %d gravacoes para o treino..." % len(alvo), flush=True)
for i, r in enumerate(alvo.itertuples()):
    Xw = windows_norm(r.filepath)
    if r.fault_type == "normal":
        assert len(Xw) == N_JAN, "%s tem %d janelas" % (r.relpath, len(Xw))
    for f in range(N_DOBRAS):
        pool = BLOCOS[f][0] if r.fault_type == "normal" else np.arange(len(Xw))
        sel = pool if len(pool) <= MAX_WIN_TRAIN else RNG_W[f].choice(pool, MAX_WIN_TRAIN, replace=False)
        TRAIN_W[f][r.relpath] = Xw[np.sort(sel)].astype(np.float16)
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(alvo), tempo()), flush=True)
print("cache de treino: %d gravacoes x %d dobras, %d janelas por dobra | %s"
      % (len(TRAIN_W[0]), N_DOBRAS, sum(len(v) for v in TRAIN_W[0].values()), tempo()))


# %% ===================================================================== T4
# Trios por ponto de operacao. A logica da rodada 9; o cache de janelas passa a
# ser o da dobra.

NORM, IMB = {}, {}
for r in alvo.itertuples():
    (NORM if r.fault_type == "normal" else IMB).setdefault(int(r.ponto), []).append(r.relpath)
CONDS = [p for p in PONTOS if p in NORM and p in IMB]
assert len(CONDS) == len(PONTOS)
print("pontos nos trios: %d | gravacoes normais %d | de desbalanceamento %d"
      % (len(CONDS), sum(len(NORM[p]) for p in CONDS), sum(len(IMB[p]) for p in CONDS)))


class TripletImbalanceDataset(Dataset):
    """Um trio por item, formado dinamicamente (mesma logica do treino original)."""

    def __init__(self, cache, steps=STEPS_PER_EPOCH):
        self.cache = cache
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        cond = CONDS[random.randrange(len(CONDS))]
        Nw = self.cache[NORM[cond][random.randrange(len(NORM[cond]))]]
        Iw = self.cache[IMB[cond][random.randrange(len(IMB[cond]))]]
        if len(Nw) < 2 or len(Iw) < 1:
            return self.__getitem__(idx)
        i1, i2 = random.sample(range(len(Nw)), 2)
        a = aug_weak(np.asarray(Nw[i1], np.float32))
        p = aug_weak(np.asarray(Nw[i2], np.float32))
        n = np.asarray(Iw[random.randrange(len(Iw))], np.float32)
        return torch.tensor(a), torch.tensor(p), torch.tensor(n)


# %% ===================================================================== T5
# Treino, uma vez por dobra. Identico ao T5 da rodada 9.

def hash_pesos(state):
    """Hash do state_dict, o mesmo do extrator (rerun_extract9, celula A2)."""
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


LOG = []
CKPTS = {}
for f in range(N_DOBRAS):
    seed = SEMENTES[f]
    print("\n=== dobra %d, semente %d | %s" % (f, seed, tempo()), flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = EmbeddingNet1D().to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    criterion = TripletMarginLoss(margin=MARGIN, p=2).to(DEVICE)
    loader = DataLoader(TripletImbalanceDataset(TRAIN_W[f]), batch_size=BATCH,
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
        LOG.append(dict(dobra=f, seed=seed, epoch=epoch + 1, loss=media,
                        minutos=round((time.time() - T_START) / 60, 1)))
        print("   epoca %02d/%02d | perda %.4f | %s" % (epoch + 1, EPOCHS, media, tempo()), flush=True)

    path = os.path.join(ROOT, "embedding_model_r10_d%d.pth" % f)
    torch.save(model.state_dict(), path)
    CKPTS[f] = dict(path=path, sha=sha_arquivo(path), pesos=hash_pesos(model.state_dict()), semente=seed)
    print("   checkpoint %s  sha %s  pesos %s" % (os.path.basename(path), CKPTS[f]["sha"], CKPTS[f]["pesos"]))
    TRAIN_W[f] = None                                # libera a memoria da dobra

# a dobra 0 contra o checkpoint da rodada 9
COMPARA_R9 = None
if ARQ["embedding_model_r9_seed42.pth"]:
    r9 = torch.load(ARQ["embedding_model_r9_seed42.pth"], map_location="cpu", weights_only=True)
    d0 = torch.load(CKPTS[0]["path"], map_location="cpu", weights_only=True)
    dif = max(float((r9[k].float() - d0[k].float()).abs().max()) for k in r9)
    COMPARA_R9 = dict(pesos_r9=hash_pesos(r9), pesos_d0=CKPTS[0]["pesos"], max_dif=dif,
                      identicos=hash_pesos(r9) == CKPTS[0]["pesos"])
    print("\ndobra 0 contra a rodada 9: %s (maior diferenca num peso %.3g)"
          % ("IDENTICOS" if COMPARA_R9["identicos"] else "diferentes", dif))

LOGDF = pd.DataFrame(LOG)
LOGDF.to_csv(os.path.join(ROOT, "treino_log.csv"), index=False)
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
for f, g in LOGDF.groupby("dobra"):
    ax.plot(g.epoch, g.loss, marker="o", ms=3.5, lw=1.4, label="dobra %d" % f)
ax.set_xlabel("epoca")
ax.set_ylabel("perda (triplet, margem %.1f)" % MARGIN)
ax.set_title("Rodada 10 -- convergencia do treino por dobra")
ax.grid(alpha=0.3, lw=0.5)
ax.set_xticks(list(LOGDF.epoch.unique()))
ax.legend(fontsize=7)
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
del TRAIN_W


# %% ===================================================================== T6
# Embeddings de TODAS as gravacoes com os cinco checkpoints. Cada gravacao e
# lida uma vez so.

def carrega_modelo(path):
    net = EmbeddingNet1D().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=True)
    net.eval()
    return net


MODELOS = {f: carrega_modelo(CKPTS[f]["path"]) for f in range(N_DOBRAS)}
ordem = meta.sort_values("relpath").reset_index(drop=True)
ACC = {f: [] for f in MODELOS}
print("embeddando %d gravacoes com %d checkpoints..." % (len(ordem), len(MODELOS)), flush=True)
for i, r in enumerate(ordem.itertuples()):
    Xw = windows_norm(r.filepath)
    for f, net in MODELOS.items():
        ACC[f].append(embed_windows_array(Xw, net).astype(EMB_DTYPE))
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(ordem), tempo()), flush=True)

for f, lst in ACC.items():
    offs = np.cumsum([0] + [len(E) for E in lst]).astype(np.int64)
    np.savez_compressed(
        os.path.join(ROOT, "mafaulda_emb_d%d.npz" % f),
        E=np.concatenate(lst), offsets=offs,
        relpath=ordem.relpath.values.astype(str), label=ordem.label.values.astype(str),
        fault_type=ordem.fault_type.values.astype(str),
        severity=ordem.severity.values.astype(str), cond=ordem.cond.values.astype(float),
        ponto=ordem.ponto.values.astype(int), ckpt=CKPTS[f]["pesos"], dobra=f,
        semente=SEMENTES[f], window=WINDOW_SIZE, stride=STRIDE, scheme="6real_win")
    print("   mafaulda_emb_d%d.npz: %d gravacoes, %d janelas" % (f, len(lst), offs[-1]))
del ACC


# %% ===================================================================== T7
# Resumo rapido de cada dobra, com os blocos dela. O numero que vale e o de
# analise_dobras.py.

def _resumo(f):
    z = np.load(os.path.join(ROOT, "mafaulda_emb_d%d.npz" % f))
    E, o = z["E"].astype(np.float64), z["offsets"]
    pt, ft = z["ponto"], z["fault_type"]
    _, en, te = BLOCOS[f]
    mod, esp, det = {}, [], []
    for i in np.flatnonzero(ft == "normal"):
        X = E[o[i]:o[i + 1]]
        Z = X[en]
        mu, var = Z.mean(0), Z.var(0, ddof=1) + 1e-12
        s = lambda Y: np.sqrt(((Y - mu) ** 2 / var).sum(1))
        T = float(np.quantile(s(Z), QUANT))
        mod[int(pt[i])] = (mu, var, T)
        esp.append(100.0 * np.mean(s(X[te]) > T))
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
    print("dobra %d: normais %d/%d saudaveis, %.2f %% das janelas normais marcadas | open-set %d/%d"
          % (f, r["normais_saudaveis"], r["normais"], r["janelas_normais_marcadas"],
             r["open_set_detectadas"], r["open_set"]))
    return r


RESUMO = {str(f): _resumo(f) for f in range(N_DOBRAS)}
json.dump(RESUMO, open(os.path.join(ROOT, "resumo.json"), "w"), indent=2)


# %% ===================================================================== T8
# Manifesto, zip, copia para o Drive e download.

manifest = dict(
    rodada=10, gerado_em=time.strftime("%Y-%m-%d %H:%M:%S"),
    dobras={str(f): dict(semente=v["semente"], arquivo=os.path.basename(v["path"]),
                         sha=v["sha"], pesos=v["pesos"]) for f, v in CKPTS.items()},
    dobra0_contra_r9=COMPARA_R9,
    treino=dict(margin=MARGIN, lr=LR, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                batch=BATCH, sementes=list(SEMENTES), max_win_train=MAX_WIN_TRAIN,
                otimizador="Adam", perda="TripletMarginLoss(p=2)"),
    trios=dict(estrategia="por ponto de operacao (gravacoes.csv)", pontos=len(CONDS),
               ancora_e_positivo="duas janelas do bloco de treino da normal do ponto, na dobra",
               negativo="janela de desbalanceamento do mesmo ponto, sorteio da dobra",
               aug="aug_weak(shift 64, amp 0.85-1.15, ruido 0.02), apos o z-score"),
    blocos="o arranjo da rodada 9 girado de %d janelas por dobra (particao.json)" % DESLOC,
    gravacoes=int(len(meta)), resumo=RESUMO, teste_local=TESTE,
    minutos=round((time.time() - T_START) / 60, 1),
    files=sorted(os.path.relpath(os.path.join(dp, f), ROOT)
                 for dp, _, fs in os.walk(ROOT) for f in fs),
)
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2)[:3000])

if not TESTE:
    import shutil
    ZIP = "/content/rodada10_treino.zip"
    os.system("cd %s && zip -q -r -0 %s ." % (ROOT, ZIP))
    print("zip %.1f MB" % (os.path.getsize(ZIP) / 1e6))
    DRIVE = "/content/drive/MyDrive"
    if os.path.isdir(DRIVE):
        shutil.copy(ZIP, os.path.join(DRIVE, os.path.basename(ZIP)))
        for v in CKPTS.values():             # os checkpoints, soltos, para o extrator
            shutil.copy(v["path"], os.path.join(DRIVE, os.path.basename(v["path"])))
        print("copiado para o Drive:", DRIVE)
    try:
        from google.colab import files as colab_files
        colab_files.download(ZIP)
    except Exception as e:
        print("download automatico indisponivel (%s); pegue pelo Drive." % e)
print("fim |", tempo())
