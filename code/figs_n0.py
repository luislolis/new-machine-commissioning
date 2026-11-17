# -*- coding: utf-8 -*-
"""
Graficos do nivel 0 no IMS (checkpoint r9), pedidos pelo Lolis em 2026-11-12:
  1. mapa s0 x n para K = 3, 7 e 14 (nivel Loos), com o desfecho de cada Set;
  2. varredura de K na referencia candidata (0 + 4 h) e na do artigo (36 + 18 h);
  3. os primeiros dias de cada Set contra a referencia de 4 h: P_r de cada rolamento e RMS (dB),
     para mostrar por que a referencia nao espera o amaciamento.

    cd code && python figs_n0.py [etiqueta] [primeiros]

Sem etiqueta, o r9, com os nomes de arquivo de sempre. Com etiqueta (por exemplo r12, da rodada 12),
os arquivos ganham o sufixo _<etiqueta>; "primeiros" faz so a figura dos primeiros dias.
"""
import json
import os
import sys

import matplotlib
import numpy as np
from scipy.stats import binom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
ARGS = sys.argv[1:]
TAG = next((a for a in ARGS if a != "primeiros"), "r9")
SO_PRIMEIROS = "primeiros" in ARGS
SUF = "" if TAG == "r9" else "_" + TAG
ROTULO = "checkpoint do artigo" if TAG == "r9" else "representação %s" % TAG
sys.argv = sys.argv[:1]
import comissionamento_n0 as c  # noqa: E402

OUT = os.path.join(H, "final_n0", "figs")
os.makedirs(OUT, exist_ok=True)
ALPHA, Q = 0.05, 0.01
S0_H = (0, 2, 4, 6, 12, 24, 36, 48, 72, 96)
N_H = (1, 2, 4, 6, 12, 18, 24, 36, 48, 72)
KS_MAPA = (3, 7, 14)
NOME = {"1st_test": "Set 1", "2nd_test": "Set 2", "4th_test": "Set 3"}
# cores: status (desfecho) e categoricas (rolamentos), paleta de referencia validada
COR_DESF = {0: "#0ca30c", 1: "#fab219", 2: "#d03b3b", 3: "#c9c8c3"}
SIMB = {0: "✓", 1: "~", 2: "✗", 3: "–"}
ROT_DESF = {0: "✓ rolamento que falhou é o primeiro",
            1: "~ saudável primeiro, junto com a falha (até 1,5 d antes do que falhou)",
            2: "✗ saudável primeiro, longe da falha",
            3: "– sem alarme ou referência não cabe"}
COR_ROL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.titlesize": 9,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

SV = c.ambiente(c.pasta(TAG))
CONT = {}


def kmin_q(nr, q, B, M):
    a = ALPHA / (B * M)
    return next((k for k in range(1, nr + 1) if binom.sf(k - 1, nr, q) <= a), nr)


def contagens(seq, b, r0, rn):
    """Janelas acima de T por snapshot, para a referencia (r0, rn); com cache."""
    ch = (seq, b, r0, rn)
    if ch not in CONT:
        E, off = SV["dados"](seq, b)[:2]
        mu, var, T = c.modelo(E, off[r0], off[r0 + rn])
        CONT[ch] = np.add.reduceat((c.pontua(E, mu, var) > T).astype(np.int16), off[:-1])
    return CONT[ch]


def loos(seq, b, s0, n):
    E, off = SV["dados"](seq, b)[:2]
    h = s0 + n // 2
    mu, var, T = c.modelo(E, off[s0], off[h])
    q = max(Q, float((c.pontua(E[off[h]:off[s0 + n]], mu, var) > T).mean()))
    return kmin_q(int(np.diff(off).max()), q, 4, SV["M_IMS"])


def roda_k(seq, b, s0, n, K, km):
    """Monitoramento do artigo (recomissionamento so nas paradas depois da referencia)."""
    E, off, day, db0, _, paradas = SV["dados"](seq, b)
    paradas = [a for a in paradas if a >= s0 + n]
    db0 = db0 - 20 * np.log10(np.mean(10.0 ** (db0[s0:s0 + n] / 20.0)))
    refs = [(s0, n, True)] + [(a, n, bool(a + n + K <= len(day) and abs(float(db0[a:a + n].mean())) <= SV["DELTA_DB"]))
                              for a in paradas]
    cur = (s0, n)
    for i, (r0, rn, ok) in enumerate(refs):
        if ok:
            cur = (r0, rn)
        ini = max(cur[0] + cur[1], paradas[i - 1] if i else 0)
        fim = paradas[i] if i < len(paradas) else len(day)
        if ini >= fim or cur[0] + cur[1] > len(day):
            continue
        j = SV["onset"](contagens(seq, b, *cur)[ini:fim] >= km, 0, K)
        if j is not None:
            return float(day[ini + j])
    return None


def desfecho(seq, s0, n, K, kms):
    day = SV["dados"](seq, 1)[2]
    if s0 + n + K > len(day):
        return 3, None, None
    fim = float(day[-1])
    al = {b: roda_k(seq, b, s0, n, K, kms[b]) for b in (1, 2, 3, 4)}
    al = {b: v for b, v in al.items() if v is not None}
    if not al:
        return 3, None, None
    p = min(al, key=al.get)
    if p in SV["FAILED"][seq]:
        return 0, fim - al[p], p
    fal = [al[b] for b in SV["FAILED"][seq] if b in al]
    if fal and min(fal) - al[p] <= 1.5:
        return 1, fim - al[p], p
    return 2, fim - al[p], p


mapa = {}
if not SO_PRIMEIROS:
    # ------------------------------------------------------------------ 1. mapa s0 x n x K
    for seq in SV["SEQS"]:
        for s0h in S0_H:
            for nh in N_H:
                s0, n = s0h * 6, nh * 6
                if s0 + n + 14 > len(SV["dados"](seq, 1)[2]):
                    for K in KS_MAPA:
                        mapa[(seq, s0h, nh, K)] = (3, None, None)
                    continue
                kms = {b: loos(seq, b, s0, n) for b in (1, 2, 3, 4)}
                for K in KS_MAPA:
                    mapa[(seq, s0h, nh, K)] = desfecho(seq, s0, n, K, kms)
        print("mapa", seq, "pronto", flush=True)

    fig, axs = plt.subplots(3, 3, figsize=(11.5, 10.5), constrained_layout=True)
    cmap = ListedColormap([COR_DESF[i] for i in range(4)])
    for i, seq in enumerate(SV["SEQS"]):
        for j, K in enumerate(KS_MAPA):
            ax = axs[i, j]
            M = np.array([[mapa[(seq, s0h, nh, K)][0] for s0h in S0_H] for nh in N_H])
            ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, origin="lower", aspect="auto")
            for a, nh in enumerate(N_H):
                for b_, s0h in enumerate(S0_H):
                    cat, lead, _ = mapa[(seq, s0h, nh, K)]
                    txt = SIMB[cat] if lead is None else "%s\n%.1f" % (SIMB[cat], lead)
                    ax.text(b_, a, txt, ha="center", va="center", fontsize=6.3, color=TINTA, linespacing=0.9)
            ax.set_xticks(range(len(S0_H)), [str(x) for x in S0_H])
            ax.set_yticks(range(len(N_H)), [str(x) for x in N_H])
            ax.set_xticks(np.arange(-0.5, len(S0_H)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(N_H)), minor=True)
            ax.grid(which="minor", color="#fcfcfb", linewidth=1.5)
            ax.tick_params(which="minor", length=0)
            if i == 2:
                ax.set_xlabel("início da referência s0 (h de aquisição)")
            if j == 0:
                ax.set_ylabel("duração n (h)")
            ax.set_title("%s, K = %d" % (NOME[seq], K), loc="left", color=TINTA)
            for s0h, nh, cor in ((0, 4, TINTA), (36, 18, TINTA2)):
                if s0h in S0_H and nh in N_H:
                    ax.add_patch(plt.Rectangle((S0_H.index(s0h) - 0.5, N_H.index(nh) - 0.5), 1, 1, fill=False,
                                               edgecolor=cor, linewidth=1.6, linestyle="-" if nh == 4 else "--"))
    hand = [plt.Rectangle((0, 0), 1, 1, color=COR_DESF[k]) for k in range(4)]
    hand += [plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=TINTA, linewidth=1.6),
             plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=TINTA2, linewidth=1.6, linestyle="--")]
    fig.legend(hand, [ROT_DESF[k] for k in range(4)] + ["0 + 4 h (teste de corrida, candidata)", "36 + 18 h (regra atual do artigo)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("IMS, " + ROTULO + ", nível Loos: desfecho por início, duração da referência e K "
                 "(número = dias antes do fim do primeiro alarme)", x=0.01, ha="left", color=TINTA)
    fig.savefig(os.path.join(OUT, "mapa_s0_n_K%s.png" % SUF), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ 2. varredura de K
    KS = range(1, 21)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True, sharey=True)
    for ax, (s0h, nh, tit) in zip(axs, ((0, 4, "referência 0 + 4 h (candidata)"), (36, 18, "referência 36 + 18 h (artigo)"))):
        for k_, seq in enumerate(SV["SEQS"]):
            s0, n = s0h * 6, nh * 6
            kms = {b: loos(seq, b, s0, n) for b in (1, 2, 3, 4)}
            res = [desfecho(seq, s0, n, K, kms) for K in KS]
            x = [K for K, r in zip(KS, res) if r[1] is not None]
            y = [r[1] for r in res if r[1] is not None]
            cor = ["#2a78d6", "#eb6834", "#1baf7a"][k_]
            ax.plot(x, y, color=cor, linewidth=1.5, label=NOME[seq])
            for K, r in zip(KS, res):
                if r[1] is not None:
                    ax.scatter(K, r[1], s=40, color=COR_DESF[r[0]], edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
                    ax.text(K, r[1] * 1.12, SIMB[r[0]], ha="center", fontsize=7, color=TINTA)
        ax.axvline(7, color=TINTA2, linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylim(1, 80)
        ax.text(7.2, 1.1, "K = 7 (artigo)", color=TINTA2, fontsize=7.5)
        ax.set_xticks([1, 3, 5, 7, 10, 14, 20])
        ax.set_xlabel("K (aquisições anômalas seguidas)")
        ax.set_title(tit, loc="left", color=TINTA)
        ax.grid(axis="y", color=GRADE, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("primeiro alarme, dias antes do fim (log)")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("IMS, " + ROTULO + ", nível Loos: efeito de K (cor e símbolo do ponto = desfecho, como no mapa)",
                 x=0.01, ha="left", color=TINTA)
    fig.savefig(os.path.join(OUT, "varredura_K%s.png" % SUF), dpi=160, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ 3. primeiros dias, referencia 0 + 4 h
HORAS = 144                                            # 6 dias de aquisicao (no Set 1 nao e calendario)
fig, axs = plt.subplots(3, 2, figsize=(11.5, 9.4), constrained_layout=True,
                        gridspec_kw=dict(width_ratios=(1.25, 1)))
for i, seq in enumerate(SV["SEQS"]):
    day = SV["dados"](seq, 1)[2]
    lim = min(HORAS * 6, len(day))
    h = np.arange(lim) / 6.0
    ax, ax2 = axs[i, 0], axs[i, 1]
    for b in (1, 2, 3, 4):
        E, off, _, db0, _, _ = SV["dados"](seq, b)
        nr = int(np.diff(off).max())
        pr = 100.0 * contagens(seq, b, 0, 24)[:lim] / nr
        lw = 2.2 if b in SV["FAILED"][seq] else 1.1
        ax.plot(h, np.convolve(pr, np.ones(6) / 6, mode="same"), color=COR_ROL[b], linewidth=lw,
                label="rolamento %d" % b)
        db = db0 - 20 * np.log10(np.mean(10.0 ** (db0[0:24] / 20.0)))
        ax2.plot(h, np.convolve(db[:lim], np.ones(6) / 6, mode="same"), color=COR_ROL[b], linewidth=lw)
        al = roda_k(seq, b, 0, 24, 7, loos(seq, b, 0, 24))
        if al is not None:
            j = int(np.searchsorted(day, al))
            if j < lim:
                ax.axvline(j / 6.0, color=COR_ROL[b], linewidth=0.9, linestyle="--")
                ax.text(j / 6.0, 97, " alarme do rol. %d" % b, color=TINTA, fontsize=7.5, va="top", rotation=90)
    for a in (ax, ax2):
        a.axvspan(0, 4, color="#c9c8c3", alpha=0.6, linewidth=0)
        for p in SV["dados"](seq, 1)[5]:
            if p < lim:
                a.axvline(p / 6.0, color=TINTA2, linewidth=0.7, linestyle=":")
        a.set_xlim(0, HORAS)
        a.set_xticks(range(0, HORAS + 1, 24))
        a.grid(axis="y", color=GRADE, linewidth=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax.axhline(100 * 7 / 39, color=TINTA2, linewidth=0.8, linestyle=":")
    ax.text(HORAS * 0.995, 100 * 7 / 39 + 1.5, "7 de 39", color=TINTA2, fontsize=7, ha="right")
    ax.set_ylim(0, 100)
    fal = " e ".join(str(b) for b in sorted(SV["FAILED"][seq]))
    ax.set_title("%s (falhou: rol. %s; linha grossa)" % (NOME[seq], fal), loc="left", color=TINTA)
    ax.set_ylabel("janelas acima do limiar (%, média de 1 h)")
    ax2.axhspan(-1.5, 1.5, color="#cde2fb", alpha=0.6, linewidth=0)
    ax2.set_title("%s: RMS de banda larga" % NOME[seq], loc="left", color=TINTA)
    ax2.set_ylabel("dB contra as 4 h iniciais (média de 1 h)")
    if i == 2:
        ax.set_xlabel("horas de aquisição desde o início")
        ax2.set_xlabel("horas de aquisição desde o início")
fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.035))
fig.suptitle("%s. Primeiras %d h de aquisição, contra a referência das 4 h iniciais (faixa cinza). Faixa azul: "
             "±1,5 dB; pontilhado vertical: paradas de 4 h ou mais" % (ROTULO.capitalize(), HORAS), x=0.01, ha="left", color=TINTA)
fig.savefig(os.path.join(OUT, "primeiros_dias%s.png" % SUF), dpi=160, bbox_inches="tight")
plt.close(fig)

if mapa:
    json.dump({"%s|%d|%d|%d" % k: v for k, v in mapa.items()}, open(os.path.join(OUT, "mapa%s.json" % SUF), "w"),
              default=str)
print("figuras em", OUT)
