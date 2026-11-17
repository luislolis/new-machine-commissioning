# -*- coding: utf-8 -*-
"""
Quanto dos 128 numeros da rede e a forma do espectro da janela?

Em janelas do MAFAULDA, ajusta uma regressao ridge de cada coordenada do embedding (rnd0, rnd1, r9)
sobre a forma do espectro da mesma janela (specn, 16 bandas por canal, deriva_r12.py) e mede o R^2
em janelas de OUTRAS gravacoes: as 1951 gravacoes sao sorteadas em duas metades, uma para ajustar
e outra para medir, e de cada metade saem 50 000 janelas. Para comparar, o mesmo com os
indicadores de forma (featn: curtose, assimetria, crista por canal).

Reporta, por representacao e preditor: R^2 total (fracao da variancia total explicada) e mediana e
quartis do R^2 das 128 coordenadas.

    cd code && python diag_rnd_spec.py
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.environ.get("MISAEL_DADOS") or H
OUT = os.path.join(H, "final_r10r11")
N_JAN, SEMENTE, LAM = 50000, 0, 1e-3
ARQ = {"specn": os.path.join(RAIZ, "rodada11", "mafaulda_emb_specn.npz"),
       "featn": os.path.join(RAIZ, "rodada11", "mafaulda_emb_featn.npz"),
       "rnd0": os.path.join(RAIZ, "rodada11", "mafaulda_emb_rnd0.npz"),
       "rnd1": os.path.join(RAIZ, "rodada11", "mafaulda_emb_rnd1.npz"),
       "r9": os.path.join(H, "rodada9", "mafaulda_emb_r9.npz")}

# ordem de referencia: a do specn; as outras sao alinhadas por relpath e conferidas janela a janela
ref = np.load(ARQ["specn"], allow_pickle=True)
REL, OFF = ref["relpath"], ref["offsets"]
nw = np.diff(OFF)
rng = np.random.default_rng(SEMENTE)
ordem = rng.permutation(len(REL))
metade = {"ajuste": np.sort(ordem[: len(REL) // 2]), "teste": np.sort(ordem[len(REL) // 2:])}
IDX = {}
for nome, recs in metade.items():
    jan = np.concatenate([np.arange(OFF[i], OFF[i + 1]) for i in recs])
    IDX[nome] = np.sort(rng.choice(jan, N_JAN, replace=False))
print("gravacoes: %d ajuste, %d teste | janelas: %d e %d (de %d e %d)"
      % (len(metade["ajuste"]), len(metade["teste"]), N_JAN, N_JAN,
         nw[metade["ajuste"]].sum(), nw[metade["teste"]].sum()), flush=True)
del ref


def amostra(nome):
    z = np.load(ARQ[nome], allow_pickle=True)
    rel, off = z["relpath"], z["offsets"]
    if np.array_equal(rel, REL):
        assert np.array_equal(off, OFF), nome
        mapa = None
    else:
        pos = {r: i for i, r in enumerate(rel)}
        assert set(pos) == set(REL), "%s: gravacoes diferentes" % nome
        assert all(off[pos[r] + 1] - off[pos[r]] == n for r, n in zip(REL, nw)), "%s: janelas diferentes" % nome
        # indice global de cada janela da ordem de referencia no arquivo deste nome
        mapa = np.concatenate([np.arange(off[pos[r]], off[pos[r] + 1]) for r in REL])
    E = z["E"]
    res = {k: np.asarray(E[v if mapa is None else mapa[v]], np.float64) for k, v in IDX.items()}
    print("   %-6s %s %s alinhamento: %s" % (nome, E.shape, E.dtype, "igual" if mapa is None else "por relpath"),
          flush=True)
    del E, z
    return res


def r2(Xa, Ya, Xt, Yt):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
    A = np.hstack([(Xa - mu) / sd, np.ones((len(Xa), 1))])
    T = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))])
    G = A.T @ A
    G[:-1, :-1] += LAM * len(A) * np.eye(A.shape[1] - 1)
    W = np.linalg.solve(G, A.T @ Ya)
    P = T @ W
    sse = ((Yt - P) ** 2).sum(0)
    sst = ((Yt - Yt.mean(0)) ** 2).sum(0)
    por = 1 - sse / sst
    return dict(r2_total=float(1 - sse.sum() / sst.sum()), mediana=float(np.median(por)),
                q1=float(np.percentile(por, 25)), q3=float(np.percentile(por, 75)),
                min=float(por.min()), max=float(por.max()))


X = {p: amostra(p) for p in ("specn", "featn")}
RES = {}
for rep in ("rnd0", "rnd1", "r9"):
    Y = amostra(rep)
    RES[rep] = {}
    for p in ("specn", "featn"):
        RES[rep][p] = r2(X[p]["ajuste"], Y["ajuste"], X[p]["teste"], Y["teste"])
        r = RES[rep][p]
        print("%-5s <- %-6s R2 total %.3f | por coordenada: mediana %.3f, quartis %.3f a %.3f, faixa %.3f a %.3f"
              % (rep, p, r["r2_total"], r["mediana"], r["q1"], r["q3"], r["min"], r["max"]), flush=True)
    del Y
json.dump(dict(janelas_por_metade=N_JAN, semente=SEMENTE, lambda_ridge=LAM,
               gravacoes_ajuste=int(len(metade["ajuste"])), gravacoes_teste=int(len(metade["teste"])),
               resultados=RES), open(os.path.join(OUT, "diag_rnd_spec.json"), "w"), indent=1)
print("salvo em final_r10r11/diag_rnd_spec.json")
