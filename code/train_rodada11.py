#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 11 -- hiperparametros e baselines de rede, no MAFAULDA.

Este .py e a FONTE (gerado por gera_rodada11.py a partir do train_rodada10). O que se abre no
Colab e train_rodada11.ipynb, gerado por `python make_notebook6.py train_rodada11`.

POR QUE
-------
A diretriz de ML da MSSP (05/12/2024) rejeita sem revisao artigos em que os hiperparametros nao
sao determinados de forma declaravel e em que nao ha comparacao com outras abordagens no mesmo
protocolo. Esta rodada produz as duas coisas, sem mudar a receita da rodada 9.

O QUE FAZ
---------
U6  ESTUDO DOS HIPERPARAMETROS. Dez dos 49 pontos de operacao, sorteados uma vez, ficam fora do
    treino. Cada configuracao treina nos outros 39 e e validada nos dez: a referencia sai do
    bloco de enrolment da normal do ponto, e decide-se o bloco de teste da normal e cada gravacao
    de desbalanceamento com o k_min do artigo. Nenhuma gravacao das quatro familias de teste
    entra. A grade varia um fator por vez em torno da receita declarada:
        margem 0.2 / 0.5 / 0.8 / 1.1; taxa 1e-4 / 3e-4 / 1e-3; dimensao 64 / 128 / 256
    (8 treinos). O resultado parcial e gravado no Drive a cada treino, e a celula retoma de onde
    parou se a sessao cair.
U7  BASELINES DE REDE. A mesma arquitetura (a) treinada como classificador normal x
    desbalanceamento, com entropia cruzada, nos mesmos lotes e pelo mesmo numero de passos
    ("ce"), e (b) sem treino, com dois sorteios de pesos ("rnd0", "rnd1").
U8  MAFAULDA INTEIRO com cada representacao: ce, rnd0, rnd1, indicadores classicos por janela
    ("feat") e espectro em bandas ("spec"). O protocolo do artigo roda depois, localmente, sobre
    cada uma, sem mudanca.

ANTES DE RODAR
--------------
Como na rodada 10: na pasta do Drive em que esta train_rodada7 (PASTA_RODADA7), deixe o
gravacoes.csv de code/mafaulda_rotacoes/.

O QUE SALVA
-----------
  rodada11_treino.zip
    hiper.csv                              a grade e a validacao de cada configuracao
    embedding_model_r11_ce.pth, _rnd0.pth, _rnd1.pth
    mafaulda_emb_ce.npz, _rnd0, _rnd1, _feat, _spec     todas as janelas das 1951 gravacoes
    particao.json, manifest.json
Os tres checkpoints tambem vao soltos para a raiz do MyDrive, para o rerun_extract11.

DEPOIS
------
rerun_extract11.ipynb (IMS, PRONOSTIA e JNU com as mesmas representacoes). Localmente:
descompactar em code/rodada11/ e code/rerun11/.

TEMPO
-----
GPU T4: ~20 min de leitura, ~15 min por treino (8 da grade + 1 do classificador, ~2 h 15) e
~50 min para ler e codificar as 1951 gravacoes: cerca de 3 h 30. O zip fica com ~1,3 GB.
"""

# %% ===================================================================== T0
# Drive, GPU e localizacao dos arquivos. Rode esta celula primeiro.

import os
import sys
import re
import json
import time
import subprocess

TESTE = os.environ.get("RODADA11_TESTE") == "1"     # ensaio local, sem Colab

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
          "rerun10", "rodada11", "rerun11", "mafaulda_emb", "ieee-phm-2012-data-challenge-dataset",
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
    print("   (o checkpoint da rodada 9 nao e usado aqui; o rerun_extract11 precisa dele)")


# %% ===================================================================== T1
# Preparo: R1-R3 da rodada 7 (constantes, rede, metadados do MAFAULDA) e as
# definicoes de R5 (leitura, janelas, aug_weak), executados do proprio codigo.

if not TESTE:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "kagglehub"], check=False)

_pre = sem_magias(secao_rodada7("train_rodada7", None, "R4"))
_pre = troca(_pre, 'ROOT = "/content/rodada7"',
             'ROOT = %r' % (os.environ["RODADA11_ROOT"] if TESTE else "/content/rodada11"))
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

N_DOBRAS = 1                # so a dobra 0, que e o arranjo de blocos da rodada 9
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


# %% ===================================================================== U5
# Funcoes desta rodada: treino parametrizado, nivel de decisao e validacao.

from scipy.stats import binom, rankdata

Q_JAN, ALPHA, M_ANO = 0.01, 0.05, 365 * 24 * 3600 / 10.0     # horizonte declarado: 10 s, um ano
assert abs((1.0 - QUANT) - Q_JAN) < 1e-9


def kmin(nr, B=1, M=M_ANO):
    """Menor k com P[Binomial(nr, q) >= k] <= alpha / (B M), como no artigo."""
    alvo = ALPHA / (B * M)
    for k in range(nr + 2):
        if binom.sf(k - 1, nr, Q_JAN) <= alvo:
            return k
    return nr + 1


assert kmin(95) == 11 and kmin(487) == 22, "k_min difere do artigo: %d, %d" % (kmin(95), kmin(487))


def semeia(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hash_pesos(state):
    """Hash do state_dict, o mesmo do extrator."""
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class Trios(Dataset):
    """Como o TripletImbalanceDataset, com a lista de pontos de operacao como parametro."""

    def __init__(self, cache, conds, steps):
        self.cache, self.conds, self.steps = cache, list(conds), steps

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        cond = self.conds[random.randrange(len(self.conds))]
        Nw = self.cache[NORM[cond][random.randrange(len(NORM[cond]))]]
        Iw = self.cache[IMB[cond][random.randrange(len(IMB[cond]))]]
        if len(Nw) < 2 or len(Iw) < 1:
            return self.__getitem__(idx)
        i1, i2 = random.sample(range(len(Nw)), 2)
        a = aug_weak(np.asarray(Nw[i1], np.float32))
        p = aug_weak(np.asarray(Nw[i2], np.float32))
        n = np.asarray(Iw[random.randrange(len(Iw))], np.float32)
        return torch.tensor(a), torch.tensor(p), torch.tensor(n)


def treina(conds, seed, margin=MARGIN, lr=LR, dim=128, perda="triplet", rotulo=""):
    """Um treino com a receita da rodada 9. Com perda="ce", a mesma rede e os mesmos lotes, com
    uma camada linear de duas classes sobre o embedding e entropia cruzada."""
    semeia(seed)
    model = EmbeddingNet1D(embed_dim=dim).to(DEVICE)
    cab = nn.Linear(dim, 2).to(DEVICE) if perda == "ce" else None
    opt = Adam(list(model.parameters()) + (list(cab.parameters()) if cab is not None else []), lr=lr)
    crit = (TripletMarginLoss(margin=margin, p=2) if perda == "triplet" else nn.CrossEntropyLoss()).to(DEVICE)
    loader = DataLoader(Trios(TRAIN_W[0], conds, STEPS_PER_EPOCH), batch_size=BATCH,
                        shuffle=True, drop_last=True, num_workers=0)
    hist = []
    model.train()
    for epoch in range(EPOCHS):
        total, nb = 0.0, 0
        for a, p, n in loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            if perda == "triplet":
                loss = crit(model(a), model(p), model(n))
            else:
                y = torch.cat([torch.zeros(len(a) + len(p)), torch.ones(len(n))]).long().to(DEVICE)
                loss = crit(cab(model(torch.cat([a, p, n]))), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        hist.append(total / max(nb, 1))
        print("   %s epoca %02d/%02d | perda %.4f | %s" % (rotulo, epoch + 1, EPOCHS, hist[-1], tempo()),
              flush=True)
    model.eval()
    return model, hist


def auc(neg, pos):
    """Area sob a curva ROC (Mann-Whitney) entre os escores das janelas normais e os da falha."""
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def valida(model, pontos):
    """Nos pontos dados: referencia no bloco de enrolment da normal; decide o bloco de teste da
    normal e cada gravacao de desbalanceamento com k_min. As familias de teste nao entram."""
    _, en, te = BLOCOS[0]
    fp_jan, fp_grav, det, pr, aucs = [], 0, 0, [], []
    for p in pontos:
        En = embed_windows_array(VALW[NORM[p][0]], model).astype(np.float64)
        mu, var = En[en].mean(0), En[en].var(0, ddof=1) + 1e-12
        s = lambda Y: np.sqrt(((Y - mu) ** 2 / var).sum(1))
        T = float(np.quantile(s(En[en]), QUANT))
        st = s(En[te])
        fp_jan.append(100.0 * np.mean(st > T))
        fp_grav += int((st > T).sum() >= kmin(len(te)))
        for rel in IMB[p]:
            si = s(embed_windows_array(VALW[rel], model).astype(np.float64))
            pr.append(100.0 * np.mean(si > T))
            det += int((si > T).sum() >= kmin(len(si)))
            aucs.append(auc(st, si))
    return dict(normais=len(pontos), normais_alarmadas=fp_grav, janelas_normais_pct=float(np.mean(fp_jan)),
                desbal=len(pr), desbal_detectadas=det, desbal_pr_medio=float(np.mean(pr)),
                desbal_pr_min=float(np.min(pr)), auc_media=float(np.mean(aucs)), auc_min=float(np.min(aucs)))


# %% ===================================================================== U6
# Estudo dos hiperparametros: treina em 39 pontos de operacao e valida nos outros 10.

rng_v = np.random.default_rng(2026)
N_VAL = 10 if not TESTE else max(1, len(CONDS) // 3)
VAL = sorted(int(p) for p in rng_v.choice(CONDS, size=N_VAL, replace=False))
TRN = [p for p in CONDS if p not in VAL]
print("pontos de validacao (%d): %s | de treino: %d" % (len(VAL), VAL, len(TRN)))

_linha = {r.relpath: r for r in meta.itertuples()}
VALW = {}
for p in VAL:
    for rel in NORM[p] + IMB[p]:
        VALW[rel] = windows_norm(_linha[rel].filepath).astype(np.float16)
print("cache de validacao: %d gravacoes, %.0f MB | %s"
      % (len(VALW), sum(v.nbytes for v in VALW.values()) / 1e6, tempo()), flush=True)

BASE_CFG = dict(margin=MARGIN, lr=LR, dim=128)
GRADE = ([dict(BASE_CFG)] + [dict(BASE_CFG, margin=m) for m in (0.2, 0.5, 1.1)]
         + [dict(BASE_CFG, lr=l) for l in (1e-4, 1e-3)] + [dict(BASE_CFG, dim=d) for d in (64, 256)])
if TESTE:
    GRADE = GRADE[:2]
DRIVE = "/content/drive/MyDrive"
PARCIAL = os.path.join(DRIVE if os.path.isdir(DRIVE) else ROOT, "rodada11_hiper_parcial.json")
HIPER = json.load(open(PARCIAL)) if os.path.exists(PARCIAL) else []
if HIPER and HIPER[0].get("validacao") != VAL:
    print("o parcial do Drive e de outra divisao de pontos; recomecando a grade")
    HIPER = []
_feitos = {json.dumps(h["cfg"], sort_keys=True) for h in HIPER}
for cfg in GRADE:
    if json.dumps(cfg, sort_keys=True) in _feitos:
        print("ja feito (parcial do Drive):", cfg)
        continue
    print("\n=== %s | %s" % (cfg, tempo()), flush=True)
    _m, _h = treina(TRN, 42, rotulo="grade", **cfg)
    HIPER.append(dict(cfg=cfg, validacao=VAL, perda_final=_h[-1], historico=_h, **valida(_m, VAL)))
    json.dump(HIPER, open(PARCIAL, "w"), indent=1)
    del _m

HDF = pd.DataFrame([dict(h["cfg"], **{k: v for k, v in h.items() if k not in ("cfg", "historico", "validacao")})
                    for h in HIPER])
HDF.to_csv(os.path.join(ROOT, "hiper.csv"), index=False)
print("\n" + HDF.to_string(index=False))
del VALW


# %% ===================================================================== U7
# Baselines de rede: a mesma arquitetura treinada como classificador, e sem treino.

N_RND = 2
CK = {}
print("\n=== classificador (entropia cruzada), todos os pontos, semente 42 | %s" % tempo(), flush=True)
CK["ce"], HIST_CE = treina(CONDS, 42, perda="ce", rotulo="ce")
for s_ in range(N_RND):
    semeia(1000 + s_)
    CK["rnd%d" % s_] = EmbeddingNet1D().to(DEVICE).eval()
CKINFO = {}
for tag, net in CK.items():
    path = os.path.join(ROOT, "embedding_model_r11_%s.pth" % tag)
    torch.save(net.state_dict(), path)
    CKINFO[tag] = dict(arquivo=os.path.basename(path), sha=sha_arquivo(path), pesos=hash_pesos(net.state_dict()))
    print("   %-5s %s  pesos %s" % (tag, CKINFO[tag]["arquivo"], CKINFO[tag]["pesos"]))
del TRAIN_W


# %% ===================================================================== U8
# MAFAULDA inteiro com cada representacao. Cada gravacao e lida uma vez.

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


def janelas_cruas(csv_path):
    """(n, 6, 1024) sem normalizar; a normalizacao de windows_norm vem depois."""
    X = read_acc_only(csv_path).to_numpy(dtype=np.float32)
    wins = [X[s:s + WINDOW_SIZE, :].T for s in range(0, X.shape[0] - WINDOW_SIZE + 1, STRIDE)]
    return np.stack(wins, axis=0) if wins else np.empty((0, 6, WINDOW_SIZE), np.float32)


def znorm_win(Xw):
    return (Xw - Xw.mean(axis=2, keepdims=True)) / (Xw.std(axis=2, keepdims=True) + 1e-8)


ordem = meta.sort_values("relpath").reset_index(drop=True)
_x = janelas_cruas(ordem.filepath.iloc[0])
assert np.array_equal(znorm_win(_x), windows_norm(ordem.filepath.iloc[0])), "janelas_cruas difere de windows_norm"
ESQ = {t: "6real_win" for t in CK}
ESQ.update(feat="6real_cru", spec="6real_cru")
ACC = {t: [] for t in ESQ}
print("codificando %d gravacoes em %d representacoes..." % (len(ordem), len(ESQ)), flush=True)
for i, r in enumerate(ordem.itertuples()):
    Xraw = janelas_cruas(r.filepath)
    Xw = znorm_win(Xraw)
    for t, net in CK.items():
        ACC[t].append(embed_windows_array(Xw, net).astype(EMB_DTYPE))
    ACC["feat"].append(feats_classicas(Xraw))
    ACC["spec"].append(espectro_bandas(Xraw).astype(EMB_DTYPE))
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(ordem), tempo()), flush=True)

for t, lst in ACC.items():
    offs = np.cumsum([0] + [len(E) for E in lst]).astype(np.int64)
    np.savez_compressed(
        os.path.join(ROOT, "mafaulda_emb_%s.npz" % t),
        E=np.concatenate(lst), offsets=offs,
        relpath=ordem.relpath.values.astype(str), label=ordem.label.values.astype(str),
        fault_type=ordem.fault_type.values.astype(str),
        severity=ordem.severity.values.astype(str), cond=ordem.cond.values.astype(float),
        ponto=ordem.ponto.values.astype(int), ckpt=CKINFO.get(t, {}).get("pesos", t),
        window=WINDOW_SIZE, stride=STRIDE, scheme=ESQ[t])
    print("   mafaulda_emb_%s.npz: %d gravacoes, %d janelas x %d" % (t, len(lst), offs[-1], lst[0].shape[1]))
del ACC


# %% ===================================================================== U9
# Manifesto, zip, copia para o Drive e download.

manifest = dict(
    rodada=11, gerado_em=time.strftime("%Y-%m-%d %H:%M:%S"),
    checkpoints=CKINFO, perda_ce=HIST_CE,
    hiper=dict(pontos_de_validacao=VAL, pontos_de_treino=len(TRN), semente=42,
               criterio="nenhuma normal de validacao alarmada; depois, mais gravacoes de "
                        "desbalanceamento detectadas; depois, maior AUC media",
               grade=[{k: v for k, v in h.items() if k != "validacao"} for h in HIPER]),
    treino=dict(margin=MARGIN, lr=LR, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch=BATCH,
                max_win_train=MAX_WIN_TRAIN, otimizador="Adam"),
    representacoes=dict(ce="rede treinada com entropia cruzada, normal x desbalanceamento",
                        rnd="rede sem treino, sementes 1000 e 1001",
                        feat="por canal: log10 RMS, log10 pico, curtose, assimetria, fator de crista",
                        spec="por canal: log10 da energia em %d bandas iguais ate fs/2" % N_ESPEC),
    gravacoes=int(len(meta)), teste_local=TESTE, minutos=round((time.time() - T_START) / 60, 1),
    files=sorted(os.path.relpath(os.path.join(dp, f), ROOT) for dp, _, fs in os.walk(ROOT) for f in fs),
)
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "hiper")}, indent=2)[:3000])

if not TESTE:
    import shutil
    ZIP = "/content/rodada11_treino.zip"
    os.system("cd %s && zip -q -r -0 %s ." % (ROOT, ZIP))
    print("zip %.1f MB" % (os.path.getsize(ZIP) / 1e6))
    if os.path.isdir(DRIVE):
        shutil.copy(ZIP, os.path.join(DRIVE, os.path.basename(ZIP)))
        for v in CKINFO.values():            # os checkpoints, soltos, para o extrator
            shutil.copy(os.path.join(ROOT, v["arquivo"]), os.path.join(DRIVE, v["arquivo"]))
        print("copiado para o Drive:", DRIVE)
    try:
        from google.colab import files as colab_files
        colab_files.download(ZIP)
    except Exception as e:
        print("download automatico indisponivel (%s); pegue pelo Drive." % e)
print("fim |", tempo())
