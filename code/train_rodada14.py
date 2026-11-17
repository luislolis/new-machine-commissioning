#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODADA 14 -- tripletos com TODAS as familias de falha do MAFAULDA (plano-rodada14.md).

Este .py e a FONTE (gerado por gera_rodada14.py a partir do train_rodada10). O que se abre no
Colab e train_rodada14.ipynb, gerado por `python make_notebook6.py train_rodada14`.

POR QUE
-------
Ate aqui a rede aprendeu a separar normal de DESBALANCEAMENTO, a unica falha nos tripletos. O
Lolis suspeita que esse padrao nao traduz a falha de rolamento do IMS. Aqui os tripletos usam as
cinco familias: desbalanceamento, desalinhamento horizontal e vertical, rolamento underhang e
overhang, nos 49 pontos de rotacao que tem normal (o ponto sem normal fica fora).

O QUE FAZ
---------
Seis treinos com a receita da rodada 9 inteira (rede, janela 1024 passo 512, z-score por janela,
aug_weak, TripletMarginLoss margem 0.8, Adam 3e-4, 12 epocas x 2000 trios, lote 64, ate 48 janelas
por gravacao), mudando so quem entra no trio:
    r14b0, r14b1, r14b2   binario: ancora e positivo da normal do ponto; negativo de uma falha do
                          ponto (familia uniforme, depois gravacao e janela)
    r14m0, r14m1, r14m2   multiclasse (normal + 5 familias): com p = 1/2 o trio binario; senao a
                          ancora e de uma familia, o positivo de OUTRA gravacao da mesma familia no
                          ponto (junta as severidades) e o negativo de outra classe do ponto
Sementes 42, 43 e 44 (indices 0, 1, 2); o binario e o multiclasse da mesma semente usam as
mesmas janelas sorteadas. Blocos das normais: os da rodada 9 (treino 0-96, enrolment 98-390,
teste 392-486). Cada checkpoint vai para a raiz do Drive assim que termina; se a sessao cair,
rode de novo: os treinos ja gravados sao pulados.

ANTES DE RODAR
--------------
Na raiz do MyDrive: train_rodada7.py e gravacoes.csv (os de code/para_o_drive_r12/). O Drive tem
de montar (a primeira celula para com instrucao se nao montar). O MAFAULDA vem do Kaggle, como nas
rodadas anteriores.

O QUE SALVA
-----------
  rodada14_treino.zip
    embedding_model_r14b0.pth ... r14m2.pth   um checkpoint por treino
    mafaulda_emb_<etiqueta>.npz               todas as janelas das 1951 gravacoes
    particao.json, trios.json, treino_log.csv, curva_perda.png, resumo.json, manifest.json
Os seis checkpoints tambem vao soltos para a raiz do MyDrive, para o rerun_extract14.

DEPOIS
------
rerun_extract14.ipynb. Localmente: descompactar em code/rodada14/.

TEMPO
-----
GPU T4: ~40 min de leitura (1919 gravacoes), ~15 min por treino (~1 h 30) e ~45 min para ler e
codificar as 1951 gravacoes com os seis checkpoints: cerca de 3 h. O zip fica com ~1,3 GB.
"""

# %% ===================================================================== T0
# Drive, GPU e localizacao dos arquivos. Rode esta celula primeiro.

import os
import sys
import re
import json
import time
import subprocess

TESTE = os.environ.get("RODADA14_TESTE") == "1"     # ensaio local, sem Colab

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
          "rerun10", "rodada11", "rerun11", "rodada12", "rerun12", "rodada13", "rerun13", "rodada14", "rerun14", "mafaulda_emb", "ieee-phm-2012-data-challenge-dataset",
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
             'ROOT = %r' % (os.environ["RODADA14_ROOT"] if TESTE else "/content/rodada14"))
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

N_DOBRAS = 1                # so o arranjo de blocos da rodada 9 (dobra 0)
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


# %% ===================================================================== V2
# Papeis da rodada 14: todas as familias de falha no treino, nos pontos que tem normal.

FAMILIAS = ("imbalance", "horizontal-misalignment", "vertical-misalignment",
            "underhang-bearing", "overhang-bearing")
SEMENTES_R14 = (42, 43, 44)
OBJETIVOS = ("b", "m")                       # binario, multiclasse
TR0, EN0, TE0 = BLOCOS[0]                    # os blocos da rodada 9
PONTOS_N = sorted(set(meta[meta.fault_type == "normal"].ponto))
meta["papel"] = np.where(meta.fault_type == "normal", "normal: treino + enrolment + teste",
                         np.where(meta.ponto.isin(PONTOS_N), "treino", "fora (ponto sem normal)"))
assert set(meta[meta.fault_type != "normal"].fault_type) <= set(FAMILIAS), sorted(set(meta.fault_type))
print("rodada 14: %d pontos com normal" % len(PONTOS_N))
print(meta.groupby(["fault_type", "papel"]).size().to_string())
json.dump(dict(blocos=dict(treino=TR0.tolist(), enrolment=EN0.tolist(), teste=TE0.tolist()),
               sementes=list(SEMENTES_R14), familias=list(FAMILIAS), pontos_com_normal=[int(p) for p in PONTOS_N],
               gravacoes={r.relpath: dict(ponto=int(r.ponto), papel=r.papel, familia=r.fault_type)
                          for r in meta.itertuples()}),
          open(os.path.join(ROOT, "particao.json"), "w"), indent=1)


# %% ===================================================================== V3
# Janelas de treino, com uma leitura so de cada gravacao: das normais, ate 48 janelas do bloco de
# treino (0-96); das falhas, ate 48 janelas da gravacao inteira. Um sorteio por semente.

RNG_W = {s: np.random.default_rng(s) for s in SEMENTES_R14}
TRAIN_W = {s: {} for s in SEMENTES_R14}
alvo = meta[(meta.fault_type == "normal") | (meta.papel == "treino")].sort_values("relpath")
print("lendo %d gravacoes para o treino..." % len(alvo), flush=True)
for i, r in enumerate(alvo.itertuples()):
    Xw = windows_norm(r.filepath)
    if r.fault_type == "normal":
        assert len(Xw) == N_JAN, "%s tem %d janelas" % (r.relpath, len(Xw))
    assert len(Xw) >= 2, "%s tem %d janelas" % (r.relpath, len(Xw))
    pool = TR0 if r.fault_type == "normal" else np.arange(len(Xw))
    for s in SEMENTES_R14:
        sel = pool if len(pool) <= MAX_WIN_TRAIN else RNG_W[s].choice(pool, MAX_WIN_TRAIN, replace=False)
        TRAIN_W[s][r.relpath] = Xw[np.sort(sel)].astype(np.float16)
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(alvo), tempo()), flush=True)
_nj = sum(len(v) for v in TRAIN_W[SEMENTES_R14[0]].values())
print("cache de treino: %d gravacoes, %d janelas por semente, %.2f GB nas %d sementes | %s"
      % (len(alvo), _nj, len(SEMENTES_R14) * _nj * 6 * WINDOW_SIZE * 2 / 1e9, len(SEMENTES_R14), tempo()))


# %% ===================================================================== V4
# Trios por ponto de rotacao, com todas as familias.

NORM, FAM = {}, {}
for r in alvo.itertuples():
    p = int(r.ponto)
    if r.fault_type == "normal":
        NORM.setdefault(p, []).append(r.relpath)
    else:
        FAM.setdefault(p, {}).setdefault(r.fault_type, []).append(r.relpath)
CONDS = [p for p in sorted(NORM) if p in FAM]
assert CONDS, "nenhum ponto com normal e falha"
_cont = pd.DataFrame([dict(ponto=p, familia=f, gravacoes=len(v)) for p in CONDS for f, v in FAM[p].items()])
print("pontos nos trios: %d | gravacoes de falha por familia (pontos em que aparece):" % len(CONDS))
print(_cont.groupby("familia").agg(gravacoes=("gravacoes", "sum"), pontos=("ponto", "nunique")).to_string())


class TripletFamilia(Dataset):
    """Um trio por item, formado dinamicamente.
    "b": ancora e positivo sao duas janelas da normal do ponto (como na rodada 9); o negativo e uma
         janela de falha do ponto, com a familia sorteada uniforme entre as presentes, depois a
         gravacao e a janela.
    "m": com p = 1/2, o trio binario; senao a ancora e uma janela de uma familia (uniforme), o
         positivo uma janela de OUTRA gravacao da mesma familia no ponto (a mesma gravacao, se for a
         unica) e o negativo uma janela de outra classe do ponto (normal ou outra familia, uniforme).
    aug_weak na ancora e no positivo, nada no negativo, como na rodada 9. self.conta guarda as classes
    (ancora/negativo) sorteadas, para conferencia."""

    def __init__(self, cache, objetivo, steps=STEPS_PER_EPOCH):
        assert objetivo in OBJETIVOS
        self.cache, self.objetivo, self.steps = cache, objetivo, steps
        self.conta = {}

    def __len__(self):
        return self.steps

    def _uma(self, rels):
        W = self.cache[rels[random.randrange(len(rels))]]
        return W[random.randrange(len(W))]

    def _anota(self, a, n):
        k = "%s/%s" % (a, n)
        self.conta[k] = self.conta.get(k, 0) + 1

    def __getitem__(self, idx):
        p = CONDS[random.randrange(len(CONDS))]
        fams = sorted(FAM[p])
        if self.objetivo == "b" or random.random() < 0.5:
            Nw = self.cache[NORM[p][random.randrange(len(NORM[p]))]]
            i1, i2 = random.sample(range(len(Nw)), 2)
            a, q = Nw[i1], Nw[i2]
            fn = fams[random.randrange(len(fams))]
            n = self._uma(FAM[p][fn])
            self._anota("normal", fn)
        else:
            fa = fams[random.randrange(len(fams))]
            recs = FAM[p][fa]
            ra = recs[random.randrange(len(recs))]
            outras = [x for x in recs if x != ra]
            Wa = self.cache[ra]
            if outras:
                a, q = Wa[random.randrange(len(Wa))], self._uma(outras)
            else:
                i1, i2 = random.sample(range(len(Wa)), 2)
                a, q = Wa[i1], Wa[i2]
            classes = ["normal"] + [f for f in fams if f != fa]
            c = classes[random.randrange(len(classes))]
            n = self._uma(NORM[p] if c == "normal" else FAM[p][c])
            self._anota(fa, c)
        a = aug_weak(np.asarray(a, np.float32))
        q = aug_weak(np.asarray(q, np.float32))
        return torch.tensor(a), torch.tensor(q), torch.tensor(np.asarray(n, np.float32))


# %% ===================================================================== V5
# Seis treinos: objetivo x semente. Cada checkpoint vai para o Drive ao terminar; ao rodar de novo,
# os treinos ja gravados no Drive sao pulados.

def hash_pesos(state):
    """Hash do state_dict, o mesmo do extrator."""
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


DRIVE = "/content/drive/MyDrive"
PERDAS_DRIVE = os.path.join(DRIVE, "rodada14_perdas.json")
TRIOS_DRIVE = os.path.join(DRIVE, "rodada14_trios.json")
LOG = json.load(open(PERDAS_DRIVE)) if (not TESTE and os.path.exists(PERDAS_DRIVE)) else []
TRIOS = json.load(open(TRIOS_DRIVE)) if (not TESTE and os.path.exists(TRIOS_DRIVE)) else {}
CKPTS = {}
for k, s in enumerate(SEMENTES_R14):
    for obj in OBJETIVOS:
        tag = "r14%s%d" % (obj, k)
        nome = "embedding_model_%s.pth" % tag
        path = os.path.join(ROOT, nome)
        no_drive = os.path.join(DRIVE, nome)
        if not TESTE and os.path.exists(no_drive) and any(l["treino"] == tag for l in LOG):
            shutil.copy(no_drive, path)
            print("\n=== %s: ja treinado (checkpoint do Drive), pulando" % tag)
        else:
            print("\n=== %s: objetivo %s, semente %d | %s" % (tag, obj, s, tempo()), flush=True)
            LOG = [l for l in LOG if l["treino"] != tag]
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            model = EmbeddingNet1D().to(DEVICE)
            opt = Adam(model.parameters(), lr=LR)
            criterion = TripletMarginLoss(margin=MARGIN, p=2).to(DEVICE)
            ds = TripletFamilia(TRAIN_W[s], obj)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
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
                LOG.append(dict(treino=tag, objetivo=obj, semente=s, epoch=epoch + 1, loss=media,
                                minutos=round((time.time() - T_START) / 60, 1)))
                print("   epoca %02d/%02d | perda %.4f | %s" % (epoch + 1, EPOCHS, media, tempo()), flush=True)
            TRIOS[tag] = dict(sorted(ds.conta.items()))
            _tot = float(sum(ds.conta.values()))
            _anc_n = sum(v for kk, v in ds.conta.items() if kk.startswith("normal/")) / _tot
            print("   trios: %d | ancora normal %.0f%% | %s" % (_tot, 100 * _anc_n,
                  ", ".join("%s %d" % kv for kv in sorted(ds.conta.items())[:8])))
            torch.save(model.state_dict(), path)
            if os.path.isdir(DRIVE) and not TESTE:
                shutil.copy(path, no_drive)
                json.dump(LOG, open(PERDAS_DRIVE, "w"), indent=1)
                json.dump(TRIOS, open(TRIOS_DRIVE, "w"), indent=1)
                print("   checkpoint copiado para o Drive:", no_drive)
            del model, opt
        st = torch.load(path, map_location="cpu", weights_only=True)
        CKPTS[tag] = dict(path=path, arquivo=nome, sha=sha_arquivo(path), pesos=hash_pesos(st),
                          objetivo=obj, semente=s)
        print("   %s  sha %s  pesos %s" % (nome, CKPTS[tag]["sha"], CKPTS[tag]["pesos"]))
    TRAIN_W[s] = None                                 # libera a memoria da semente
del TRAIN_W
json.dump(TRIOS, open(os.path.join(ROOT, "trios.json"), "w"), indent=1)
assert len(set(v["pesos"] for v in CKPTS.values())) == len(CKPTS), "dois treinos deram os mesmos pesos"

LOGDF = pd.DataFrame(LOG)
LOGDF.to_csv(os.path.join(ROOT, "treino_log.csv"), index=False)
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
for t, g in LOGDF.groupby("treino"):
    ax.plot(g.epoch, g.loss, marker="o", ms=3.5, lw=1.4, ls="-" if t[3] == "b" else "--", label=t)
ax.set_xlabel("epoca")
ax.set_ylabel("perda (triplet, margem %.1f)" % MARGIN)
ax.set_title("Rodada 14 -- tripletos com todas as familias")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(fontsize=7, ncol=2)
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


# %% ===================================================================== V6
# Embeddings de TODAS as gravacoes com os seis checkpoints. Cada gravacao e lida uma vez so.

def carrega_modelo(path):
    net = EmbeddingNet1D().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=True)
    net.eval()
    return net


MODELOS = {t: carrega_modelo(v["path"]) for t, v in CKPTS.items()}
ordem = meta.sort_values("relpath").reset_index(drop=True)
ACC = {t: [] for t in MODELOS}
print("embeddando %d gravacoes com %d checkpoints..." % (len(ordem), len(MODELOS)), flush=True)
for i, r in enumerate(ordem.itertuples()):
    Xw = windows_norm(r.filepath)
    for t, net in MODELOS.items():
        ACC[t].append(embed_windows_array(Xw, net).astype(EMB_DTYPE))
    if (i + 1) % 100 == 0:
        print("   %4d/%4d | %s" % (i + 1, len(ordem), tempo()), flush=True)

for t, lst in ACC.items():
    offs = np.cumsum([0] + [len(E) for E in lst]).astype(np.int64)
    np.savez_compressed(
        os.path.join(ROOT, "mafaulda_emb_%s.npz" % t),
        E=np.concatenate(lst), offsets=offs,
        relpath=ordem.relpath.values.astype(str), label=ordem.label.values.astype(str),
        fault_type=ordem.fault_type.values.astype(str), papel=ordem.papel.values.astype(str),
        severity=ordem.severity.values.astype(str), cond=ordem.cond.values.astype(float),
        ponto=ordem.ponto.values.astype(int), ckpt=CKPTS[t]["pesos"], objetivo=CKPTS[t]["objetivo"],
        semente=CKPTS[t]["semente"], window=WINDOW_SIZE, stride=STRIDE, scheme="6real_win")
    print("   mafaulda_emb_%s.npz: %d gravacoes, %d janelas" % (t, len(lst), offs[-1]))
    ACC[t] = None
del ACC


# %% ===================================================================== V7
# Resumo rapido de cada treino. As falhas foram TODAS vistas no treino: so a especificidade dos
# blocos de teste das normais vale como teste. O numero que vale e o de analise_r14.py.

def _resumo(t):
    z = np.load(os.path.join(ROOT, "mafaulda_emb_%s.npz" % t))
    E, o = z["E"].astype(np.float64), z["offsets"]
    pt, ft = z["ponto"], z["fault_type"]
    mod, esp, det = {}, [], {}
    for i in np.flatnonzero(ft == "normal"):
        X = E[o[i]:o[i + 1]]
        Z = X[EN0]
        mu, var = Z.mean(0), Z.var(0, ddof=1) + 1e-12
        sc = lambda Y: np.sqrt(((Y - mu) ** 2 / var).sum(1))
        T = float(np.quantile(sc(Z), QUANT))
        mod[int(pt[i])] = (mu, var, T)
        esp.append(100.0 * np.mean(sc(X[TE0]) > T))
    for i in np.flatnonzero(ft != "normal"):
        if int(pt[i]) not in mod:
            continue
        mu, var, T = mod[int(pt[i])]
        X = E[o[i]:o[i + 1]]
        det.setdefault(str(ft[i]), []).append(100.0 * np.mean(np.sqrt(((X - mu) ** 2 / var).sum(1)) > T))
    esp = np.array(esp)
    r = dict(normais_saudaveis=int((esp < DECISION_LEVEL).sum()), normais=int(len(esp)),
             janelas_normais_marcadas=float(esp.mean()),
             familias_vistas_no_treino={f: "%d/%d" % (int((np.array(v) >= DECISION_LEVEL).sum()), len(v))
                                        for f, v in sorted(det.items())})
    print("%s: normais %d/%d saudaveis, %.2f %% das janelas normais marcadas | falhas (vistas): %s"
          % (t, r["normais_saudaveis"], r["normais"], r["janelas_normais_marcadas"],
             ", ".join("%s %s" % kv for kv in r["familias_vistas_no_treino"].items())))
    return r


RESUMO = {t: _resumo(t) for t in CKPTS}
json.dump(RESUMO, open(os.path.join(ROOT, "resumo.json"), "w"), indent=2)


# %% ===================================================================== V8
# Manifesto, zip, copia para o Drive e download.

manifest = dict(
    rodada=14, gerado_em=time.strftime("%Y-%m-%d %H:%M:%S"),
    checkpoints={t: {k: v for k, v in c.items() if k != "path"} for t, c in CKPTS.items()},
    treino=dict(margin=MARGIN, lr=LR, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch=BATCH,
                sementes=list(SEMENTES_R14), max_win_train=MAX_WIN_TRAIN, otimizador="Adam",
                perda="TripletMarginLoss(p=2)", familias=list(FAMILIAS)),
    trios=dict(estrategia="por ponto de operacao (gravacoes.csv)", pontos=len(CONDS),
               b="ancora e positivo da normal do ponto; negativo: familia uniforme do ponto, depois gravacao e janela",
               m="p = 1/2 o trio b; senao ancora de uma familia, positivo de outra gravacao da mesma familia, "
                 "negativo de outra classe do ponto (uniforme)",
               aug="aug_weak(shift 64, amp 0.85-1.15, ruido 0.02) na ancora e no positivo",
               contagem=TRIOS),
    blocos="os da rodada 9: treino %d-%d, enrolment %d-%d, teste %d-%d"
           % (TR0[0], TR0[-1], EN0[0], EN0[-1], TE0[0], TE0[-1]),
    gravacoes=int(len(meta)), resumo=RESUMO, teste_local=TESTE,
    minutos=round((time.time() - T_START) / 60, 1),
    files=sorted(os.path.relpath(os.path.join(dp, f), ROOT) for dp, _, fs in os.walk(ROOT) for f in fs),
)
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "trios")}, indent=2)[:3000])

if not TESTE:
    ZIP = "/content/rodada14_treino.zip"
    os.system("cd %s && zip -q -r -0 %s ." % (ROOT, ZIP))
    print("zip %.1f MB" % (os.path.getsize(ZIP) / 1e6))
    if os.path.isdir(DRIVE):
        shutil.copy(ZIP, os.path.join(DRIVE, os.path.basename(ZIP)))
        for v in CKPTS.values():            # os checkpoints, soltos, para o extrator
            shutil.copy(v["path"], os.path.join(DRIVE, v["arquivo"]))
        print("copiado para o Drive:", DRIVE)
    try:
        from google.colab import files as colab_files
        colab_files.download(ZIP)
    except Exception as e:
        print("download automatico indisponivel (%s); pegue pelo Drive." % e)
print("fim |", tempo())
