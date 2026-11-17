"""
Rotacao no MAFAULDA: a que ponto de operacao nominal pertence cada gravacao.

A documentacao do MAFAULDA diz que cada pasta de falha (classe e severidade)
varre "the same 49 values employed in the normal operation case". O nome do
arquivo, porem, e a rotacao MEDIDA, numa grade de 0,2048 Hz, e por isso quase
nunca coincide com a da gravacao normal do mesmo ponto. Aqui cada pasta e
alinhada a varredura normal por programacao dinamica monotona: a k-esima
rotacao da pasta vai para um ponto nominal, na ordem, sem cruzar e sem repetir
(repeticao so com penalidade), e um ponto nominal pode ficar sem gravacao. O
desvio que sobra, rotacao medida menos rotacao da normal do mesmo ponto, e o
que se classifica.

Classes de caso, por gravacao
    E  exato     desvio 0
    J  jitter    1 ou 2 passos de 0,2048 Hz (ate 0,41 Hz)
    G  grande    3 passos ou mais
    R  reordenado: o ponto do alinhamento nao e a normal mais proxima
    V  sem nominal: o ponto nao tem gravacao normal (lacuna da varredura normal)
    D  duplicado: a pasta tem duas gravacoes no mesmo ponto
(R, V e D sao marcas que se somam a E/J/G.)

Saidas: mafaulda_rotacoes/gravacoes.csv, pontos.csv, resumo.txt

    cd code && python analise_rotacoes_mafaulda.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "mafaulda_rotacoes")
os.makedirs(OUT, exist_ok=True)

PASSO = 0.2048          # resolucao do nome do arquivo, Hz
PEN_DUP = 1.0           # Hz: custo de pôr duas gravacoes da pasta no mesmo ponto

z = np.load(os.path.join(H, "rodada7", "mafaulda_feats.npz"), allow_pickle=True)
meta = pd.DataFrame(dict(relpath=z["relpath"], pasta=z["label"], classe=z["fault_type"],
                         severidade=z["severity"], f=z["cond"].astype(float),
                         split_r7=z["split"]))
meta["pasta"] = meta.pasta.str.replace("mafaulda/", "", regex=False)

# --------------------------------------------------- pontos nominais
nor = np.sort(meta[meta.classe == "normal"].f.values)
passo_med = np.median(np.diff(nor))
pontos = [dict(f_nom=f, normal=True) for f in nor]
# lacunas da varredura normal: um ponto sem normal, no meio das gravacoes de falha
falhas = meta[meta.classe != "normal"].f.values
for a, b in zip(nor[:-1], nor[1:]):
    if b - a > 1.5 * passo_med:
        dentro = falhas[(falhas > a + 0.4) & (falhas < b - 0.4)]
        fv = float(np.median(dentro)) if len(dentro) else (a + b) / 2
        pontos.append(dict(f_nom=fv, normal=False))
P = pd.DataFrame(pontos).sort_values("f_nom").reset_index(drop=True)
P["ponto"] = np.arange(1, len(P) + 1)
y = P.f_nom.values


def alinha(x):
    """x em ordem crescente -> indice do ponto de cada x (DP monotona)."""
    m, n = len(x), len(y)
    C = np.abs(x[:, None] - y[None, :])
    D = np.full((m, n), np.inf)
    T = np.zeros((m, n), dtype=int)          # ponto da gravacao anterior
    D[0] = C[0]
    for i in range(1, m):
        # melhor prefixo estritamente antes de j
        best = np.minimum.accumulate(D[i - 1])
        arg = np.zeros(n, dtype=int)
        cur = 0
        for j in range(n):
            if D[i - 1, j] <= D[i - 1, cur]:
                cur = j
            arg[j] = cur
        antes = np.r_[np.inf, best[:-1]]
        arg_antes = np.r_[0, arg[:-1]]
        mesmo = D[i - 1] + PEN_DUP
        usa_mesmo = mesmo < antes
        D[i] = np.where(usa_mesmo, mesmo, antes) + C[i]
        T[i] = np.where(usa_mesmo, np.arange(n), arg_antes)
    j = int(np.argmin(D[-1]))
    idx = [j]
    for i in range(m - 1, 0, -1):
        j = T[i, j]
        idx.append(j)
    return np.array(idx[::-1])


linhas = []
for pasta, g in meta.groupby("pasta"):
    g = g.sort_values("f")
    x = g.f.values
    if (g.classe == "normal").all():
        j = np.searchsorted(y, x)          # as normais sao os proprios pontos
    else:
        j = alinha(x)
    prox = np.argmin(np.abs(x[:, None] - y[None, :]), axis=1)
    dup = pd.Series(j).duplicated(keep=False).values
    for r, jj, pp, dd in zip(g.itertuples(), j, prox, dup):
        d = r.f - y[jj]
        linhas.append(dict(relpath=r.relpath, pasta=pasta, classe=r.classe,
                           severidade=r.severidade, f=r.f, ponto=int(P.ponto[jj]),
                           f_nom=y[jj], tem_normal=bool(P.normal[jj]),
                           desvio_hz=d, desvio_passos=int(round(d / PASSO)),
                           desvio_pct=100.0 * d / y[jj],
                           reordenado=bool(jj != pp), duplicado=bool(dd),
                           split_r7=r.split_r7))
G = pd.DataFrame(linhas)
a = G.desvio_passos.abs()
G["caso"] = np.where(a == 0, "E", np.where(a <= 2, "J", "G"))
G["marcas"] = (G.reordenado.map({True: "R", False: ""}) + (~G.tem_normal).map({True: "V", False: ""})
               + G.duplicado.map({True: "D", False: ""}))
G.to_csv(os.path.join(OUT, "gravacoes.csv"), index=False)

# --------------------------------------------------- relatorio
saida = []


def p(s=""):
    print(s)
    saida.append(s)


F = G[G.classe != "normal"]
p("MAFAULDA: %d gravacoes, %d pastas de falha, %d pontos nominais (%d com normal)"
  % (len(G), F.pasta.nunique(), len(P), P.normal.sum()))
p("passo da varredura normal: mediana %.3f Hz, de %.3f a %.3f Hz"
  % (passo_med, np.diff(nor).min(), np.diff(nor).max()))
for r in P[~P.normal].itertuples():
    p("ponto sem normal: %d, em %.3f Hz (entre as normais %.2f e %.2f Hz)"
      % (r.ponto, r.f_nom, y[r.ponto - 2], y[r.ponto]))

p("\n1. Casos, nas %d gravacoes de falha" % len(F))
tab = F.groupby("caso").size().reindex(["E", "J", "G"]).fillna(0).astype(int)
for c, n in tab.items():
    p("   %s  %4d  (%.1f %%)" % (c, n, 100.0 * n / len(F)))
for mk, nome in (("R", "reordenado"), ("V", "sem nominal"), ("D", "duplicado")):
    n = F.marcas.str.contains(mk).sum()
    p("   %s  %4d  %s" % (mk, n, nome))
p("   desvio |Hz|: mediana %.3f, p90 %.3f, max %.3f"
  % (F.desvio_hz.abs().median(), F.desvio_hz.abs().quantile(0.9), F.desvio_hz.abs().max()))

p("\n2. Por classe")
t = F.groupby("classe").agg(n=("f", "size"), E=("caso", lambda s: (s == "E").sum()),
                            J=("caso", lambda s: (s == "J").sum()),
                            G=("caso", lambda s: (s == "G").sum()),
                            R=("marcas", lambda s: s.str.contains("R").sum()),
                            V=("marcas", lambda s: s.str.contains("V").sum()),
                            D=("marcas", lambda s: s.str.contains("D").sum()),
                            desvio_medio_passos=("desvio_passos", "mean"),
                            negativos=("desvio_passos", lambda s: (s < 0).sum()),
                            positivos=("desvio_passos", lambda s: (s > 0).sum()))
p(t.round(2).to_string())

p("\n3. A falha desloca a rotacao? desvio assinado por severidade (passos de 0,2048 Hz)")


def num(s):
    try:
        return float("".join(ch for ch in s if ch.isdigit() or ch == "."))
    except ValueError:
        return np.nan


for cl in ("imbalance", "horizontal-misalignment", "vertical-misalignment",
           "underhang-bearing", "overhang-bearing"):
    g = F[F.classe == cl].copy()
    g["sev"] = g.pasta.str.split("/").str[-1].map(num)
    m = g.groupby("pasta").desvio_passos.mean()
    rho, pv = spearmanr(g.sev, g.desvio_passos)
    p("   %-24s %s | Spearman(severidade, desvio) = %+.2f (p = %.2f)"
      % (cl, "  ".join("%s %+.2f" % (k.split("/", 1)[1], v) for k, v in m.items()), rho, pv))
rho, pv = spearmanr(F.f_nom, F.desvio_passos)
p("   desvio contra a rotacao nominal, todas as falhas: Spearman %+.2f (p = %.2f)" % (rho, pv))

p("\n4. Cobertura: pontos nominais varridos por pasta")
cob = F.groupby("pasta").agg(n=("f", "size"), pontos=("ponto", "nunique"),
                            f_max=("f_nom", "max"))
for r in cob.itertuples():
    p("   %-34s %3d gravacoes, %2d pontos, ate %.1f Hz" % (r.Index, r.n, r.pontos, r.f_max))

p("\n5. Diferenca para o pareamento da rodada 7 (rotacao normal mais proxima)")
T7 = F[F.split_r7 == "test"]
p("   particao de teste da rodada 7: %d falhas, %d com ponto nominal diferente do mais proximo, "
  "%d num ponto sem normal" % (len(T7), T7.reordenado.sum(), (~T7.tem_normal).sum()))

pt = G.groupby(["ponto", "f_nom", "tem_normal"]).agg(gravacoes=("f", "size"),
                                                   pastas=("pasta", "nunique")).reset_index()
pt.to_csv(os.path.join(OUT, "pontos.csv"), index=False)
open(os.path.join(OUT, "resumo.txt"), "w", encoding="utf-8").write("\n".join(saida) + "\n")
p("\nsaidas em %s" % OUT)
