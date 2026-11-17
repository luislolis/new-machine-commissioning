# -*- coding: utf-8 -*-
"""Figuras da versao de submissao (regra B, checkpoint r9), em ingles, para o artigo.

    cd code && python figuras_v10.py [ims] [pron] [jnu] [maf] [first] [resp] [sweep] [scale]

Sem argumento, todas. Saida: ../figs/<nome>_v10.png (e copia em final_v10/figs/).
Dados: final_v10/numeros.json (numeros finais), r9 (embeddings, via apoio_v10/comissionamento_n0),
final_n0/figs/mapa.json, final_n0/escala_nivel.csv, rodada12/resposta_mafaulda.csv.
Estilo: figuras_v8.py (fontes, eixos), paleta de figs_n0.py. Fundo branco, 300 dpi.
"""
import gc
import json
import os
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]

FIGS = os.path.normpath(os.path.join(H, "..", "figs"))
COPIA = os.path.join(H, "final_v10", "figs")
os.makedirs(COPIA, exist_ok=True)
NUM = json.load(open(os.path.join(H, "final_v10", "numeros.json"), encoding="utf-8"))["r9"]

COR_ROL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
COR_DESF = {0: "#0ca30c", 1: "#fab219", 2: "#d03b3b", 3: "#c9c8c3"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e6e5e0"
FAIXA = "0.86"
ALARME = "#d03b3b"
NOME = {"1st_test": "Set 1", "2nd_test": "Set 2", "4th_test": "Set 3"}
SEQS = ("1st_test", "2nd_test", "4th_test")
GAP_CURVA = 2 / 24
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "savefig.facecolor": "white"})

_SV = {}


def sv():
    """O ambiente do r9 (SV), carregado uma vez."""
    if "SV" not in _SV:
        import apoio_v10 as ap
        import comissionamento_n0 as c
        _SV["ap"], _SV["c"] = ap, c
        _SV["SV"] = c.ambiente(ap.pasta_v10("r9"))
    return _SV["SV"]


def salva(fig, nome):
    p = os.path.join(FIGS, nome)
    fig.savefig(p, dpi=300)
    plt.close(fig)
    shutil.copy(p, os.path.join(COPIA, nome))
    print("gravada", p, flush=True)


def gapped(x, y, max_gap_days=GAP_CURVA):
    x, y = np.asarray(x, float), np.asarray(y, float)
    cut = np.where(np.diff(x) > max_gap_days)[0] + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)


# ------------------------------------------------------------------------------ 1. IMS
def segmentos(S, b, n=24, s0=0):
    """Replica ap.linha_do_tempo e devolve tambem a referencia vigente em cada aquisicao.
    Devolve (alarmes [indices de onset], uso [lista de (inicio, ref)], janelas_ref [(r0, n)], kmin por ref)."""
    SVv, K = S.SV, 7
    E, off, day, db0, _, paradas = SVv["dados"](S.seq, b)
    N = len(day)
    base_db = lambda r0: 20 * np.log10(np.mean(10.0 ** (db0[r0:r0 + n] / 20.0)))
    # como ap.linha_do_tempo (corrigido em 2026-11-16): a base do aceite e o k_min sao os do
    # comissionamento (inicial ou depois de inspecao); numa parada aceita so mu, sigma2 e T mudam
    cur, pos, al = (s0, n), s0 + n, []
    base, km = base_db(s0), S.loos(b, s0, n)
    uso, jan = [(s0 + n, cur, km)], [(s0, n, "initial")]
    stops = [a for a in paradas if a >= s0 + n]
    while pos < N:
        prox = [a for a in stops if a > pos]
        fimseg = prox[0] if prox else N
        j = SVv["onset"](S.contagens(b, *cur)[pos:fimseg] >= km, 0, K)
        if j is not None:
            a = pos + j
            al.append(a)
            r0 = a + K
            if r0 + n + K > N:
                break
            cur, pos, base, km = (r0, n), r0 + n, base_db(r0), S.loos(b, r0, n)
            uso.append((pos, cur, km))
            jan.append((r0, n, "alarm"))
            continue
        if not prox:
            break
        a = prox[0]
        if a + n + K <= N and abs(float((db0[a:a + n] - base).mean())) <= 1.5:
            cur, pos = (a, n), a + n
            uso.append((pos, cur, km))
            jan.append((a, n, "stop"))
        else:
            pos = a
    return al, uso, jan


def fig_ims():
    SV = sv()
    import regra_rms as rr
    paradas_ref = {}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6))
    leg_dados = None
    for ax, seq in zip(axes, SEQS):
        S = rr.Set(SV, seq)
        dd = SV["dados"](seq, 1)[2]
        fim = float(dd[-1])
        N = len(dd)
        falhou = NUM["ims"][seq]["falhou"]
        paradas = SV["dados"](seq, 1)[5]
        hb, JAN = {}, set()
        for b in (1, 2, 3, 4):
            E, off = SV["dados"](seq, b)[:2]
            nr = int(np.diff(off).max())
            al, uso, jan = segmentos(S, b)
            # confere com a linha do tempo do numeros.json (dias antes do fim)
            lt = [round(fim - float(dd[a]), 2) for a in al]
            ref_json = NUM["ims"][seq]["rol"][str(b)]["linha_do_tempo"]
            assert lt == ref_json, (seq, b, lt, ref_json)
            pr = np.full(N, np.nan)
            nivel = np.full(N, np.nan)
            for (ini, cur, km_), nxt in zip(uso, [u[0] for u in uso[1:]] + [N]):
                pr[ini:nxt] = 100.0 * S.contagens(b, *cur)[ini:nxt] / nr
                nivel[ini:nxt] = 100.0 * km_ / nr
            for r0, n_, _ in jan:                                     # dentro da propria referencia: nada
                pr[r0:r0 + n_] = np.nan
                nivel[r0:r0 + n_] = np.nan
            y = pd.Series(pr).rolling(12, center=True, min_periods=1).mean().where(~np.isnan(pr))
            f = b in falhou
            hb[b], = ax.plot(*gapped(dd, y.values), color=COR_ROL[b], lw=1.3 if f else 0.65,
                             alpha=1.0 if f else 0.85, zorder=3 if f else 2)
            ax.plot(*gapped(dd, nivel), color=COR_ROL[b], lw=0.55, ls=(0, (1.5, 1.5)), alpha=0.9, zorder=1)
            ymk = 104.5 + 5.0 * (b - 1)
            for a in al:
                ax.plot(dd[a], ymk, marker="v", ms=4.2, color=COR_ROL[b], mec="white", mew=0.4, clip_on=False, zorder=5)
                ax.axvline(dd[a], color=COR_ROL[b], lw=0.5, alpha=0.4, zorder=0)
            for r0, n_, tipo in jan:
                if tipo == "initial":
                    ax.axvspan(dd[r0], dd[r0 + n_ - 1], color=FAIXA, lw=0, zorder=0)
                else:
                    ax.add_patch(plt.Rectangle((dd[r0], -11.4 + 2.5 * (b - 1)), dd[min(r0 + n_, N) - 1] - dd[r0], 2.2,
                                               color=COR_ROL[b], lw=0, alpha=0.75, zorder=1))
        for p in paradas:
            ax.axvline(dd[p], color="0.45", lw=0.6, ls=":", zorder=0)
        ax.set_ylim(-12.5, 102)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("$P_r$ (%)")
        ax.set_xlim(-0.01 * fim, 1.01 * fim)
        ax.set_title("IMS %s (failed: bearing%s %s)" % (NOME[seq], "s" if len(falhou) > 1 else "",
                                                       " and ".join(str(b) for b in falhou)), loc="left", pad=22)
        ax.grid(axis="y", color=GRADE, lw=0.5)
        leg_dados = hb
    axes[-1].set_xlabel("Time since the first acquisition (days)")
    fig.tight_layout(rect=(0, 0.115, 1, 1), h_pad=1.8)
    hs = [Line2D([], [], color=COR_ROL[b], lw=1.3 if b in (3, 4) else 0.65) for b in (1, 2, 3, 4)]
    ls_ = ["Bearing %d" % b for b in (1, 2, 3, 4)]
    hs += [Line2D([], [], color="0.4", lw=1.3), Patch(color=FAIXA, lw=0),
           Patch(color="0.4", alpha=0.75, lw=0),
           Line2D([], [], color="0.4", lw=0.55, ls=(0, (1.5, 1.5))),
           Line2D([], [], color="0.4", marker="v", ls="", ms=4.2),
           Line2D([], [], color="0.45", lw=0.6, ls=":")]
    ls_ += ["thick line: bearing that failed in that Set", "initial reference (first 4 h)",
            "later references, one row per bearing (after an alarm or a stop)",
            "level $k_{\\min}/N_r$ in force", "alarm onset (one row per bearing)", "stop of 4 h or more"]
    fig.legend(hs, ls_, loc="lower center", ncol=2, frameon=False, handlelength=2.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.0))
    salva(fig, "imsResults_v10.png")


# ------------------------------------------------------------------------------ 2. PRONOSTIA
def _pron_dados():
    """Contagens por aquisicao dos 17 rolamentos (referencia 0..29), conferidas contra numeros.json."""
    if "pron" in _SV:
        return _SV["pron"]
    SV = sv()
    c = _SV["c"]
    PR = NUM["pron"]["rolamentos"]
    dados = {}
    for bn in SV["PRON"]:
        E, off, _ = c.pron_dados(SV, bn)
        mu, var, T = c.modelo(E, off[0], off[30])
        k = np.add.reduceat((c.pontua(E, mu, var) > T).astype(np.int64), off[:-1])
        nr = int(np.diff(off).max())
        r = PR[bn]
        i = SV["onset"](k >= r["kmin"], 30, 7)
        assert i == r["onset"], (bn, i, r["onset"])
        dados[bn] = dict(k=k, nr=nr, **r)
        del E
    gc.collect()
    _SV["pron"] = dados
    return dados


def fig_pron_all():
    """Os 17 rolamentos numa figura so (enxugamento de 2026-11-16): uma linha por condicao, eixo em horas."""
    dados = _pron_dados()
    linhas = [[b for b in dados if dados[b]["cond"] == cnd] for cnd in (1, 2, 3)]
    ncols = max(len(l) for l in linhas)
    fig, axes = plt.subplots(3, ncols, figsize=(7.2, 4.6), squeeze=False)
    rot = {1: "Condition 1\n1800 rpm, 4000 N", 2: "Condition 2\n1650 rpm, 4200 N", 3: "Condition 3\n1500 rpm, 5000 N"}
    tem_corte = False
    for li, uns in enumerate(linhas):
        for ci in range(ncols):
            ax = axes[li, ci]
            if ci >= len(uns):
                ax.axis("off")
                continue
            d = dados[uns[ci]]
            n = len(d["k"])
            x = np.arange(n) * 10 / 3600.0
            y = 100.0 * d["k"] / d["nr"]
            ax.axvspan(0, 30 * 10 / 3600.0, color=FAIXA, lw=0)
            ax.plot(x, y, color="0.8", lw=0.4)
            ax.plot(x, pd.Series(y).rolling(30, center=True, min_periods=1).mean(), color=COR_ROL[1], lw=0.9)
            ax.axhline(100.0 * d["kmin"] / d["nr"], color="0.5", ls="--", lw=0.5)
            if d["onset"] is not None:
                ax.axvline(x[d["onset"]], color=ALARME, ls="--", lw=0.8)
            if d["truncado_em"]:
                tem_corte = True
                ax.axvline(d["truncado_em"] * 10 / 3600.0, color="0.3", ls="-.", lw=0.6)
            ax.set_ylim(-4, 104)
            ax.set_yticks([0, 50, 100])
            ax.set_xlim(0, x[-1])
            ax.tick_params(labelsize=6, length=2, pad=1)
            ax.set_title(uns[ci].replace("Bearing", "B"), pad=1.5, fontsize=7)
            if ci == 0:
                ax.set_ylabel(rot[li + 1] + "\n$P_r$ (%)", fontsize=6.5)
            else:
                ax.set_yticklabels([])
    fig.supxlabel("Time from the start of the run (h)", fontsize=7)
    fig.tight_layout(w_pad=0.4, h_pad=0.6)
    hs = [Line2D([], [], color="0.8", lw=0.8), Line2D([], [], color=COR_ROL[1], lw=0.9), Patch(color=FAIXA, lw=0),
          Line2D([], [], color="0.5", ls="--", lw=0.5), Line2D([], [], color=ALARME, ls="--", lw=0.8)]
    ls_ = ["$P_r$ per acquisition (steps of 25%)", "30-acquisition moving average",
           "reference (30 acquisitions, 3 s of signal)", "decision level (4 of 4)", "alarm onset"]
    if tem_corte:
        hs.append(Line2D([], [], color="0.3", ls="-.", lw=0.6))
        ls_.append("end of the truncated test file")
    vazio = axes[2, len(linhas[2])].get_position()
    fim = axes[2, ncols - 1].get_position()
    fig.legend(hs, ls_, loc="center", bbox_to_anchor=((vazio.x0 + fim.x1) / 2, vazio.y0 + vazio.height / 2),
               bbox_transform=fig.transFigure, ncol=2, frameon=False, handlelength=2.0, fontsize=6.5,
               columnspacing=1.2, labelspacing=0.4)
    salva(fig, "pronostiaResults_all_v10.png")


def fig_pron():
    dados = _pron_dados()
    for conds, ncols, fname in (((1,), 4, "pronostiaResults_v10.png"), ((2, 3), 4, "pronostiaResults_c23_v10.png")):
        uns = [b for b in dados if dados[b]["cond"] in conds]
        nrows = int(np.ceil(len(uns) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.75 * nrows + (0.2 if nrows == 1 else 0)), squeeze=False)
        tem_corte = False
        for ax, bn in zip(axes.ravel(), uns):
            d = dados[bn]
            n = len(d["k"])
            x = np.arange(n) * 10 / 60.0
            y = 100.0 * d["k"] / d["nr"]
            ax.axvspan(0, 30 * 10 / 60.0, color=FAIXA, lw=0)
            ax.plot(x, y, color="0.75", lw=0.5)
            ax.plot(x, pd.Series(y).rolling(30, center=True, min_periods=1).mean(), color=COR_ROL[1], lw=1.0)
            ax.axhline(100.0 * d["kmin"] / d["nr"], color="0.5", ls="--", lw=0.6)
            ax.set_ylim(-4, 104)
            ax.set_yticks([0, 25, 50, 75, 100])
            if d["onset"] is not None:
                ax.axvline(x[d["onset"]], color=ALARME, ls="--", lw=0.9)
            if d["truncado_em"]:
                tem_corte = True
                ax.axvline(d["truncado_em"] * 10 / 60.0, color="0.3", ls="-.", lw=0.7)
            ax.set_title(bn.replace("_", "\\_") if False else bn, pad=2)
        vazios = list(axes.ravel()[len(uns):])
        for ax in vazios:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("min")
        for ax in axes[:, 0]:
            ax.set_ylabel("$P_r$ (%)")
        fig.tight_layout()
        hs = [Line2D([], [], color="0.75", lw=0.8), Line2D([], [], color=COR_ROL[1], lw=1.0), Patch(color=FAIXA, lw=0),
              Line2D([], [], color="0.5", ls="--", lw=0.6), Line2D([], [], color=ALARME, ls="--", lw=0.9)]
        ls_ = ["$P_r$ per acquisition (steps of 25%)", "30-acquisition moving average", "reference (30 acquisitions)",
               "decision level (4 of 4)", "alarm onset"]
        if tem_corte:
            hs.append(Line2D([], [], color="0.3", ls="-.", lw=0.7))
            ls_.append("end of the truncated test file")
        p0 = vazios[0].get_position()
        ym = p0.y0 + p0.height / 2
        if len(vazios) == 1:
            fig.legend(hs, ls_, loc="center right", bbox_to_anchor=(0.998, ym), bbox_transform=fig.transFigure,
                       ncol=1, frameon=False, handlelength=2.0, fontsize=6.5, borderaxespad=0.0)
        else:
            fig.legend(hs, ls_, loc="center left", bbox_to_anchor=(p0.x0 - 0.04, ym), bbox_transform=fig.transFigure,
                       ncol=1, frameon=False, handlelength=2.6)
        salva(fig, fname)


# ------------------------------------------------------------------------------ 3. JNU
def fig_jnu():
    SV = sv()
    c = _SV["c"]
    J = NUM["jnu"]
    ordem = ["normal", "inner", "ball", "outer"]
    rot = {"normal": "N (held out)", "inner": "IR", "ball": "RE", "outer": "OR"}
    M = np.array([[J["por_rpm"][r][o]["pct_marcadas"] for r in ("600", "800", "1000")] for o in ordem])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3), ["%s rpm" % r for r in ("600", "800", "1000")])
    ax.set_yticks(range(len(ordem)), [rot[o] for o in ordem])
    for i in range(len(ordem)):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, "%.1f" % v, ha="center", va="center", fontsize=7, color="white" if v < 60 else "black")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="anomalous windows (%)")
    fig.tight_layout()
    salva(fig, "jnuHeatmap_v10.png")

    EJ, OJ, LAB, RPM = SV["EJ"], SV["OJ"], SV["LABJ"], SV["RPMJ"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, r in zip(axes, ("600", "800", "1000")):
        a, n = J["por_rpm"][r]["trava"]
        assert (a, n) == tuple(J["trava"][r])
        i_n = [i for i in range(12) if LAB[i] == "normal" and RPM[i] == int(r)][0]
        En = EJ[OJ[i_n]:OJ[i_n + 1]]
        mu, var, T = c.modelo(En, a, a + n)
        assert abs(T - J["por_rpm"][r]["T"]) < 1e-6
        dat, nom = [], []
        for o in ordem:
            i = [k for k in range(12) if LAB[k] == o and RPM[k] == int(r)][0]
            X = En[len(En) // 2:] if o == "normal" else EJ[OJ[i]:OJ[i + 1]]
            dat.append(c.pontua(X, mu, var))
            nom.append(rot[o].split()[0])
            kk = int((dat[-1] > T).sum())
            assert kk == J["por_rpm"][r][o]["k"], (r, o, kk)
        ax.boxplot(dat, whis=1.5, showfliers=False, labels=nom, widths=0.6,
                   medianprops=dict(color=TINTA), boxprops=dict(color=TINTA2), whiskerprops=dict(color=TINTA2),
                   capprops=dict(color=TINTA2))
        ax.axhline(T, color=ALARME, ls="--", lw=0.8)
        ax.set_title("%s rpm" % r)
        ax.grid(axis="y", color=GRADE, lw=0.5)
    axes[0].set_ylabel("Mahalanobis score")
    axes[-1].legend([Line2D([], [], color=ALARME, ls="--", lw=0.8)], ["window threshold $T$"], loc="upper left",
                    frameon=False)
    fig.tight_layout()
    salva(fig, "jnuBoxplot_v10.png")


# ------------------------------------------------------------------------------ 4. MAFAULDA
def fig_maf():
    import apoio_v10 as ap
    res, R = ap.mafaulda("r9")
    R = R.copy()
    R["pct"] = 100.0 * R.k / R.n
    R["kmin_loos"] = [ap.kmin_q(int(n), q) for n, q in zip(R.n, R.q)]
    R["niv"] = 100.0 * R.kmin_loos / R.n
    L = NUM["mafaulda"]["loos"]
    FAM = ["normal", "imbalance", "horizontal-misalignment", "vertical-misalignment", "underhang-bearing", "overhang-bearing"]
    ROT = {"normal": "Normal", "imbalance": "Imbalance", "horizontal-misalignment": "Horiz. misalign.",
           "vertical-misalignment": "Vert. misalign.", "underhang-bearing": "Underhang bearing",
           "overhang-bearing": "Overhang bearing"}
    pareado = R.grupo.isin(["normal", "desbal", "open-set"])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    dados = [R[pareado & (R.classe == f)].pct.values for f in FAM]
    ax.boxplot(dados, whis=1.5, showfliers=True, labels=[ROT[f] for f in FAM], widths=0.6,
               medianprops=dict(color=TINTA), boxprops=dict(color=TINTA2), whiskerprops=dict(color=TINTA2),
               capprops=dict(color=TINTA2), flierprops=dict(ms=2, mfc="0.4", mec="none"))
    for i, f in enumerate(FAM, start=1):
        niv = R[pareado & (R.classe == f)].niv
        print("nivel Loos %-24s mediana %.1f %% faixa %.1f-%.1f" % (f, niv.median(), niv.min(), niv.max()))
        ax.hlines(niv.median(), i - 0.42, i + 0.42, color=ALARME, ls="--", lw=0.9,
                  label="decision level $k_{\\min}/N_r$ (median of the recordings)" if i == 1 else None)
        if niv.max() - niv.min() > 0.5:
            ax.vlines(i + 0.42, niv.min(), niv.max(), color=ALARME, lw=0.9, label="range of the level over the recordings" if i == 1 else None)
    ax.set_ylabel("Anomalous windows (%)")
    ax.set_ylim(-4, 104)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, borderaxespad=0.2, columnspacing=1.2)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.tight_layout()
    salva(fig, "boxplotMafaulda_v10.png")

    av = R[R.grupo.isin(["normal", "open-set"])]
    LIN = ["normal", "horizontal-misalignment", "vertical-misalignment", "underhang-bearing", "overhang-bearing"]
    cmc = np.array([[int((~av[av.classe == c].alarme_loos).sum()), int(av[av.classe == c].alarme_loos.sum())] for c in LIN])
    tn, fp, fn, tp = int(cmc[0, 0]), int(cmc[0, 1]), int(cmc[1:, 0].sum()), int(cmc[1:, 1].sum())
    assert (tp, fn, fp, tn) == (L["tp"], L["fn"], L["fp"], L["tn"]), (tp, fn, fp, tn)
    frac = cmc / cmc.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1], ["Normal", "Fault"])
    ax.set_yticks(range(len(LIN)), ["Normal (test blocks)"] + [ROT[c] for c in LIN[1:]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True condition")
    ax.spines[:].set_visible(False)
    for i in range(len(LIN)):
        for j in range(2):
            ax.text(j, i, "%d (%.1f%%)" % (cmc[i, j], 100 * frac[i, j]), ha="center", va="center", fontsize=7.5,
                    color="white" if frac[i, j] > 0.5 else "black")
    fig.tight_layout()
    salva(fig, "confusioMatrix_v10.png")
    print("matriz por condicao:", cmc.tolist())


# ------------------------------------------------------------------------------ 5. primeiros dias
def fig_first():
    SV = sv()
    import regra_rms as rr
    HORAS = 144
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.0), gridspec_kw=dict(width_ratios=(1.15, 1)))
    for i, seq in enumerate(SEQS):
        S = rr.Set(SV, seq)
        day = SV["dados"](seq, 1)[2]
        lim = min(HORAS * 6, len(day))
        h = np.arange(lim) / 6.0
        ax, ax2 = axs[i, 0], axs[i, 1]
        falhou = NUM["ims"][seq]["falhou"]
        for b in (1, 2, 3, 4):
            E, off, _, db0, _, _ = SV["dados"](seq, b)
            nr = int(np.diff(off).max())
            pr = 100.0 * S.contagens(b, 0, 24)[:lim] / nr
            lw = 1.9 if b in falhou else 0.9
            ax.plot(h, np.convolve(pr, np.ones(6) / 6, mode="same"), color=COR_ROL[b], linewidth=lw)
            ax.axhline(100.0 * NUM["ims"][seq]["rol"][str(b)]["kmin"] / nr, color=COR_ROL[b], lw=0.5,
                       ls=(0, (1.5, 1.5)))
            db = db0 - 20 * np.log10(np.mean(10.0 ** (db0[0:24] / 20.0)))
            ax2.plot(h, np.convolve(db[:lim], np.ones(6) / 6, mode="same"), color=COR_ROL[b], linewidth=lw)
            ad = NUM["ims"][seq]["rol"][str(b)]["alarme_dia"]
            if ad is not None:
                j = int(np.searchsorted(day, ad))
                if j < lim:
                    ax.axvline(j / 6.0, color=COR_ROL[b], linewidth=0.8, linestyle="--")
                    ax.text(j / 6.0 - 1.0, 97, "first alarm, bearing %d" % b, color=TINTA, fontsize=7, va="top",
                            ha="right", rotation=90)
        for a in (ax, ax2):
            a.axvspan(0, 4, color="#c9c8c3", alpha=0.7, linewidth=0)
            for p in SV["dados"](seq, 1)[5]:
                if p < lim:
                    a.axvline(p / 6.0, color=TINTA2, linewidth=0.6, linestyle=":")
            a.set_xlim(0, HORAS)
            a.set_xticks(range(0, HORAS + 1, 24))
            a.grid(axis="y", color=GRADE, linewidth=0.5)
        ax.set_ylim(0, 100)
        ax.set_title("%s (failed: bearing%s %s)" % (NOME[seq], "s" if len(falhou) > 1 else "",
                                                     " and ".join(str(b) for b in falhou)), loc="left")
        ax.set_ylabel("$P_r$ (%, 1 h mean)")
        ax2.axhspan(-1.5, 1.5, color="#cde2fb", alpha=0.6, linewidth=0)
        ax2.set_title("%s: broadband RMS" % NOME[seq], loc="left")
        ax2.set_ylabel("dB re first 4 h (1 h mean)")
        if i == 2:
            ax.set_xlabel("acquisition hours since the start")
            ax2.set_xlabel("acquisition hours since the start")
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=1.2, w_pad=1.2)
    hs = [Line2D([], [], color=COR_ROL[b], lw=1.9 if b in (3, 4) else 0.9) for b in (1, 2, 3, 4)]
    ls_ = ["Bearing %d" % b for b in (1, 2, 3, 4)]
    hs += [Line2D([], [], color="0.4", lw=1.9), Patch(color="#c9c8c3", alpha=0.7, lw=0),
           Line2D([], [], color="0.4", lw=0.5, ls=(0, (1.5, 1.5))), Patch(color="#cde2fb", alpha=0.6, lw=0),
           Line2D([], [], color=TINTA2, lw=0.6, ls=":")]
    ls_ += ["thick: bearing that failed", "reference (first 4 h)", "level $k_{\\min}/N_r$ (left)",
            "$\\pm$1.5 dB (right)", "stop of 4 h or more"]
    fig.legend(hs, ls_, loc="lower center", ncol=5, frameon=False, handlelength=2.0, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    salva(fig, "firstDays_v10.png")


# ------------------------------------------------------------------------------ 7. varredura (K = 7)
def fig_sweep():
    S0_H = (0, 2, 4, 6, 12, 24, 36, 48, 72, 96)
    N_H = (1, 2, 4, 6, 12, 18, 24, 36, 48, 72)
    mapa = json.load(open(os.path.join(H, "final_n0", "figs", "mapa.json")))
    SIMB = {0: "\u2713", 1: "~", 2: "\u2717", 3: "\u2013"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    cmap = ListedColormap([COR_DESF[i] for i in range(4)])
    for ax, seq in zip(axs, SEQS):
        M = np.array([[mapa["%s|%d|%d|7" % (seq, s0h, nh)][0] for s0h in S0_H] for nh in N_H])
        ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, origin="lower", aspect="auto", alpha=0.9)
        for a in range(len(N_H)):
            for b_ in range(len(S0_H)):
                ax.text(b_, a, SIMB[int(M[a, b_])], ha="center", va="center", fontsize=6.5, color=TINTA)
        ax.set_xticks(range(len(S0_H)), [str(x) for x in S0_H], fontsize=6.5)
        ax.set_yticks(range(len(N_H)), [str(x) for x in N_H], fontsize=6.5)
        ax.set_xticks(np.arange(-0.5, len(S0_H)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(N_H)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        ax.spines[:].set_visible(False)
        ax.set_xlabel("reference start $s_0$ (h)")
        ax.set_title(NOME[seq], loc="left")
        ax.add_patch(plt.Rectangle((S0_H.index(0) - 0.5, N_H.index(4) - 0.5), 1, 1, fill=False, edgecolor=TINTA, lw=1.8, zorder=6, clip_on=False))
        if ax is not axs[0]:
            ax.tick_params(axis="y", left=False)
    axs[0].set_ylabel("reference length $n$ (h)")
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6)
    hand = [Patch(color=COR_DESF[k]) for k in range(4)] + [Patch(fill=False, edgecolor=TINTA, lw=1.8)]
    lab = ["\u2713 the bearing that failed alarms first", "~ healthy bearing first, within 1.5 d of the one that failed",
           "\u2717 healthy bearing first, far from the failure", "\u2013 no alarm, or the reference does not fit",
           "0 + 4 h, the reference of this work"]
    fig.legend(hand, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.4,
               columnspacing=1.5)
    salva(fig, "sweep_s0_n_v10.png")
    # confere a celula 0 + 4 h com o numeros.json
    for seq in SEQS:
        print(NOME[seq], "celula 0+4h:", mapa["%s|0|4|7" % seq], "json:", NUM["ims"][seq]["desfecho"])


ALL = dict(ims=fig_ims, pron=fig_pron, pronall=fig_pron_all, jnu=fig_jnu, maf=fig_maf, first=fig_first, sweep=fig_sweep)
if __name__ == "__main__":
    for k in (ARGS or list(ALL)):
        ALL[k]()
    print("pronto")
