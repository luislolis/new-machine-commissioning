"""
Numeros e figuras do artigo -- PROTOCOLO FINAL (rodada 7).

O que mudou em relacao ao analise_final_v3.py
---------------------------------------------
1. CHECKPOINT da rodada 7 (max pooling k=4 s=2 apos cada ReLU; 1024 -> 512 ->
   256 -> 128; campo receptivo 78 amostras). Embeddings em rerun7/.
2. Regra unica de canais (repeticao ciclica) e z-score POR JANELA em todas as
   bases -- a variante rep_win.
3. VARIANCIA POR COORDENADA no lugar da covariancia cheia: o modelo saudavel
   guarda um vetor de 128 variancias, nao uma matriz 128x128. Bem posto sem
   regularizacao pesada (128 parametros contra 120 a 4212 janelas), e a
   ablacao mostra que os alarmes nao mudam.
4. RE-COMISSIONAMENTO no IMS apos parada documentada: parada >= 4 h, nova
   referencia de 18 h, aceita se o RMS da unidade estiver a menos de 1,5 dB
   da referencia original. Criterio absoluto, externo ao detector.
5. Decisao SO pelo embedding -- sem canal de severidade.

Protocolo (inalterado no resto)
-------------------------------
Escore     s(z) = sqrt( sum_i (z_i - mu_i)^2 / (var_i + eps) ), eps = 1e-12.
Limiar     T = Q(1-q) dos escores da referencia, q = 0,01.
Snapshot   anomalo se k >= k_min, o menor k com Binom_sf(k, N_r, q) <= alfa/(B M).
Alarme     K = 10 snapshots anomalos seguidos; onset = o primeiro deles.

Saidas: final_v4/*.json, *.csv, *.txt e ../figs/*_v4.png

    cd code && python analise_final_v4.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.join(H, "rerun7")
FIG = os.path.join(H, "..", "figs")
OUT = os.path.join(H, "final_v4")
os.makedirs(OUT, exist_ok=True)

Q, ALPHA, K, EPS = 0.01, 0.05, 10, 1e-12
QW = 1.0 - Q
# Horizonte de monitoramento declarado em TEMPO e convertido pelo intervalo de
# aquisicao de cada sistema: M = T / dt. O intervalo e propriedade da aquisicao,
# conhecida no comissionamento; o numero de snapshots que a maquina produz antes
# de falhar, nao. Assim nenhuma base informa o proprio criterio -- antes o JNU
# usava M = 12, o tamanho do corpus.
T_HORIZONTE = 365 * 24 * 3600              # um ano de operacao continua
DT_IMS, DT_PRON, DT_JNU = 600.0, 10.0, 10.0
M_IMS = T_HORIZONTE / DT_IMS               # 5.26e4
M_PRON = T_HORIZONTE / DT_PRON             # 3.15e6
M_JNU = T_HORIZONTE / DT_JNU               # 3.15e6
# Janela de validacao declarada em TEMPO e convertida por dt, como o horizonte.
# 10 min: a regra seleciona a mesma referencia que com 5 min nas duas bases
# (varredura em final_v4/sens_V.csv), e acima de 30 min nao existe referencia
# admissivel no PRONOSTIA, porque V passaria a cobrir boa parte da corrida mais
# curta, que tem 230 aquisicoes, 38 min.
V_TEMPO = 600.0
VA_IMS = max(1, int(round(V_TEMPO / DT_IMS)))      # 1 snapshot
VA_PRON = max(1, int(round(V_TEMPO / DT_PRON)))    # 60 aquisicoes
# Piso do enrolamento declarado em TEMPO DE SINAL: 3 s, o menor valor redondo
# que entrega 1/q janelas na mais lenta das tres aquisicoes (IMS, 38,1 janelas
# por segundo de sinal; o ponto exato e 2,62 s). Em janelas: 114, 120 e 293.
PISO_SINAL = 3.0
DUR_IMS, DUR_PRON = 1.024, 0.100                   # uma aquisicao, em segundos
GAP_H, N_RE, DELTA_DB = 4.0, 108, 1.5      # re-comissionamento
IMS_S0, IMS_N = 216, 108                   # 36 h + 18 h
PR_S0, PR_N = 18, 30                       # 3 min + 5 min (selecao da regra)
SEQS = ("1st_test", "2nd_test", "4th_test")
FAILED = {"1st_test": {3: "inner race", 4: "rolling element"},
          "2nd_test": {1: "outer race"},
          "4th_test": {3: "outer race"}}
CH = {"1st_test": 2, "2nd_test": 1, "4th_test": 1}
# nomes da documentacao do IMS (Set No. 1, 2, 3), usados nos titulos das figuras
NOME_IMS = {"1st_test": "Set 1", "2nd_test": "Set 2", "4th_test": "Set 3"}
PRON = ["Bearing%d_%d" % (c, i) for c, n in ((1, 7), (2, 7), (3, 3)) for i in range(1, n + 1)]
NUM = {}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
BCOL = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd"}


# --------------------------------------------------------------- protocolo
def sf(k, n, q=Q):
    return float(binom.sf(k - 1, n, q))


def kmin(nr, M, B):
    return next(k for k in range(1, nr + 1) if sf(k, nr) <= ALPHA / (B * M))


def fit(Ef):
    """Modelo saudavel: media e VETOR de variancias por coordenada."""
    Ef = Ef.astype(np.float64)
    return Ef.mean(axis=0), Ef.var(axis=0, ddof=1) + EPS


def scores(E, mu, var, chunk=40000):
    out = np.empty(len(E))
    for a in range(0, len(E), chunk):
        D = E[a:a + chunk].astype(np.float64) - mu
        out[a:a + chunk] = np.sqrt((D * D / var).sum(axis=1))
    return out


def onset(flags, start, K_=K):
    run = 0
    for j in range(start, len(flags)):
        run = run + 1 if flags[j] else 0
        if run >= K_:
            return j - K_ + 1
    return None


def gapped(x, y, max_gap_days=2 / 24):
    x, y = np.asarray(x, float), np.asarray(y, float)
    cut = np.where(np.diff(x) > max_gap_days)[0] + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)


# --------------------------------------------------------------------- IMS
def ims_load(seq, b):
    z = np.load(os.path.join(R7, "ims_rep_win", "ims_%s_b%d_emb.npz" % (seq, b)))
    t = pd.to_datetime(pd.Series(z["files"]), format="%Y.%m.%d.%H.%M.%S")
    return (z["E"].astype(np.float64), z["offsets"],
            ((t - t.iloc[0]).dt.total_seconds() / 86400).values)


def ims_db(seq, b):
    """RMS em dB sobre a referencia original, para o criterio de aceitacao."""
    F = np.load(os.path.join(H, "rerun4_artifacts", "ims_%s_feat.npz" % seq))
    fn = list(F["feat_names"])
    c = CH[seq]
    rms = F["feats"][:, (b - 1) * c:b * c, fn.index("rms")].mean(axis=1)
    return 20 * np.log10(rms / rms[IMS_S0:IMS_S0 + IMS_N].mean())


def ims_monitor(seq, b):
    """Percorre os trechos entre paradas, re-comissionando quando aceito.

    Devolve P_r por snapshot, a mascara de anomalia, o dia do alarme, e a lista
    de referencias em vigor (indice inicial, duracao, aceita, dB medido).
    """
    E, off, day = ims_load(seq, b)
    nr = int(np.diff(off).max())
    km = kmin(nr, M_IMS, 4)
    db0 = ims_db(seq, b)
    paradas = list(np.where(np.diff(day) * 24 >= GAP_H)[0] + 1)
    refs = [(IMS_S0, IMS_N, True, 0.0)]
    for a in paradas:
        cabe = a + N_RE + K <= len(day)
        nivel = float(db0[a:a + N_RE].mean()) if cabe else float("nan")
        refs.append((a, N_RE, bool(cabe and abs(nivel) <= DELTA_DB), nivel))
    k = np.zeros(len(day), np.int64)
    z = np.full(len(day), np.nan)
    mon = np.zeros(len(day), bool)
    bandas = []
    cur, alarme, isnap = (IMS_S0, IMS_N), None, None
    for i, (r0, rn, ok, _) in enumerate(refs):
        if ok:
            cur = (r0, rn)
            bandas.append((cur[0], cur[0] + cur[1]))
        ini = max(cur[0] + cur[1], paradas[i - 1] if i else 0)
        fim = paradas[i] if i < len(paradas) else len(day)
        if ini >= fim:
            continue
        mu, var = fit(E[off[cur[0]]:off[cur[0] + cur[1]]])
        s = scores(E, mu, var)
        T = float(np.quantile(s[off[cur[0]]:off[cur[0] + cur[1]]], QW))
        kk = np.add.reduceat((s > T).astype(np.int64), off[:-1])
        msc = np.add.reduceat(s, off[:-1]) / np.diff(off)
        ref = msc[cur[0]:cur[0] + cur[1]]
        k[ini:fim] = kk[ini:fim]
        z[ini:fim] = ((msc - ref.mean()) / (ref.std(ddof=1) + 1e-12))[ini:fim]
        mon[ini:fim] = True
        j = onset((kk >= km)[ini:fim], 0)
        if j is not None and alarme is None:
            alarme, isnap = day[ini + j], ini + j
    return dict(day=day, k=k, z=z, mon=mon, nr=nr, kmin=km, onset=alarme, isnap=isnap,
                refs=refs, bandas=bandas)


print("=" * 78)
print("IMS -- rodada 7, referencia 36 h + 18 h, variancia por coordenada, re-comissionamento")
print("=" * 78)
ims_rows, IMS = [], {}
for seq in SEQS:
    IMS[seq] = {b: ims_monitor(seq, b) for b in (1, 2, 3, 4)}
    fim = IMS[seq][1]["day"][-1]
    aceitos = sum(1 for r in IMS[seq][1]["refs"][1:] if r[2])
    print("\n%s | fim dia %.2f | k_min %d de %d | paradas >= %.0f h: %d, re-comissionadas: %d"
          % (seq, fim, IMS[seq][1]["kmin"], IMS[seq][1]["nr"], GAP_H,
             len(IMS[seq][1]["refs"]) - 1, aceitos))
    for b in (1, 2, 3, 4):
        d = IMS[seq][b]
        lead = fim - d["onset"] if d["onset"] is not None else None
        ims_rows.append(dict(seq=seq, bearing=b, failed=b in FAILED[seq],
                             defect=FAILED[seq].get(b, ""), snapshot=d["isnap"],
                             onset_day=d["onset"], lead_days=lead, end_day=fim,
                             kmin=d["kmin"], nr=d["nr"]))
        print("   rol %d%-16s %s" % (b, " (%s)" % FAILED[seq][b] if b in FAILED[seq] else "",
                                     "alarme dia %6.2f | antecedencia %5.2f d | snapshot %d"
                                     % (d["onset"], lead, d["isnap"]) if d["onset"] is not None
                                     else "nenhum alarme"))
IMSDF = pd.DataFrame(ims_rows)
IMSDF.to_csv(os.path.join(OUT, "ims_onsets.csv"), index=False)

for seq in SEQS:
    g = IMSDF[(IMSDF.seq == seq) & IMSDF.onset_day.notna()].sort_values("onset_day")
    if g.empty:
        continue
    prim = g.iloc[0]
    falha = g[g.failed]
    NUM["ims_%s_first" % seq] = dict(
        bearing=int(prim.bearing), failed=bool(prim.failed), day=float(prim.onset_day),
        lead=float(prim.lead_days),
        failed_lead=float(falha.iloc[0].lead_days) if len(falha) else None,
        healthy_before=[int(b) for b in g[~g.failed].bearing
                        if len(falha) and g.set_index("bearing").onset_day[b] < falha.iloc[0].onset_day])


# -------------------------------------------------------------- PRONOSTIA
def pron_load(bn):
    z = np.load(os.path.join(R7, "A", "pronostia", "pronostia_rep_win_%s_emb.npz" % bn))
    return (z["E"].astype(np.float64), z["offsets"], int(z["n_truncated"]),
            int(z["condition"]), str(z["subset"]))


def temp_no(bn, i):
    """Temperatura no inicio e no onset, pelo metodo de analise_final_v3.py."""
    F = np.load(os.path.join(H, "rerun4_artifacts", "pronostia_%s_feat.npz" % bn))
    tv, tt, t_acc = F["temp_mean"], F["temp_t0_seconds"], F["t0_seconds"]
    if not len(tv):
        return None, None
    if i is None:
        return float(tv[0]), None
    t_rel = (tt - t_acc[0]) / 3600
    t_rel = np.where(t_rel < -1, t_rel + 24, t_rel)
    return float(tv[0]), float(tv[np.argmin(np.abs(t_rel - i * 10 / 3600.0))])


print("\n" + "=" * 78)
print("PRONOSTIA -- 17 rolamentos, referencia 3 min + 5 min")
print("=" * 78)
pr_rows, PR = [], {}
for bn in PRON:
    E, off, ntr, cond, sub = pron_load(bn)
    ns = len(off) - 1
    nr = int(np.diff(off).max())
    km = kmin(nr, M_PRON, 1)
    mu, var = fit(E[off[PR_S0]:off[PR_S0 + PR_N]])
    s = scores(E, mu, var)
    T = float(np.quantile(s[off[PR_S0]:off[PR_S0 + PR_N]], QW))
    k = np.add.reduceat((s > T).astype(np.int64), off[:-1])
    i = onset(k >= km, PR_S0 + PR_N)
    PR[bn] = dict(k=k, nr=nr, kmin=km, n=ns, onset=i, ntr=ntr, cond=cond)
    t0, ton = temp_no(bn, i)
    pr_rows.append(dict(unit=bn, cond=cond, subset=sub, snapshots=ns, n_truncated=ntr,
                        kmin=km, nr=nr, onset=i,
                        lead_h=None if i is None else (ns - i) * 10 / 3600.0,
                        remaining_pct=None if i is None else 100.0 * (ns - i) / ns,
                        in_validation=bool(i is not None and i - (PR_S0 + PR_N) < VA_PRON),
                        before_truncation=bool(ntr > 0 and i is not None and i < ntr),
                        cut_h=None if not (ntr > 0 and i is not None and i < ntr) else (ntr - i) * 10 / 3600.0,
                        temp_first=t0, temp_onset=ton))
PRDF = pd.DataFrame(pr_rows)
PRDF.to_csv(os.path.join(OUT, "pronostia_onsets.csv"), index=False)
ok = PRDF.onset.notna()
print("   alarme antes da falha em %d de %d | antecedencia %.2f a %.2f h (mediana %.2f)"
      % (ok.sum(), len(PRDF), PRDF[ok].lead_h.min(), PRDF[ok].lead_h.max(), PRDF[ok].lead_h.median()))
print("   fracao da corrida restante: %.2f a %.2f %% (mediana %.2f)"
      % (PRDF[ok].remaining_pct.min(), PRDF[ok].remaining_pct.max(), PRDF[ok].remaining_pct.median()))
print("   dentro da janela de validacao: %s" % (", ".join(PRDF[PRDF.in_validation].unit) or "nenhum"))
c1 = PRDF[PRDF.cond == 1]
c23 = PRDF[PRDF.cond != 1]
for nome, g in (("condicao 1 (calibracao)", c1), ("condicoes 2 e 3 (retidas)", c23)):
    gg = g[g.onset.notna()]
    print("   %-26s %d de %d | restante %.2f a %.2f %% (mediana %.2f)"
          % (nome, len(gg), len(g), gg.remaining_pct.min(), gg.remaining_pct.max(),
             gg.remaining_pct.median()))
tt_ = PRDF[ok & PRDF.temp_onset.notna()]
print("   temperatura no onset (%d com registro): %s"
      % (len(tt_), ", ".join("%s %.0f" % (r.unit, r.temp_onset) for r in tt_.itertuples())))
print("   temperatura no inicio: %.0f a %.0f" % (PRDF.temp_first.min(), PRDF.temp_first.max()))
rho, p = spearmanr(PRDF[ok].snapshots, PRDF[ok].remaining_pct)
print("   Spearman duracao x fracao restante: rho = %.2f (p = %.3f)" % (rho, p))
NUM["pronostia"] = dict(fired=int(ok.sum()), n=len(PRDF),
                        remaining_min=float(PRDF[ok].remaining_pct.min()),
                        remaining_max=float(PRDF[ok].remaining_pct.max()),
                        remaining_median=float(PRDF[ok].remaining_pct.median()),
                        lead_min_h=float(PRDF[ok].lead_h.min()),
                        lead_max_h=float(PRDF[ok].lead_h.max()),
                        in_validation=list(PRDF[PRDF.in_validation].unit),
                        spearman_rho=float(rho), spearman_p=float(p))


# -------------------------------------------------------------------- JNU
print("\n" + "=" * 78)
print("JNU -- referencia de 3 s desde o inicio da gravacao normal")
print("=" * 78)
zj = np.load(os.path.join(R7, "A", "jnu", "jnu_rep_win_emb.npz"))
Ej, oj = zj["E"].astype(np.float64), zj["offsets"]
lab, rpm, arq = list(zj["labels"]), list(zj["rpm"]), list(zj["files"])
WPS = 2931 / 30.0
a_j, n_j = 0, int(round(PISO_SINAL * WPS))  # 0 s + 3 s (piso de sinal)
jnu_rows = []
for r in (600, 800, 1000):
    i_n = [i for i in range(12) if lab[i] == "normal" and rpm[i] == r][0]
    En = Ej[oj[i_n]:oj[i_n + 1]]
    mu, var = fit(En[a_j:a_j + n_j])
    T = float(np.quantile(scores(En[a_j:a_j + n_j], mu, var), QW))
    for i in range(12):
        if rpm[i] != r:
            continue
        X = En[len(En) // 2:] if i == i_n else Ej[oj[i]:oj[i + 1]]
        sc = scores(X, mu, var)
        c = int((sc > T).sum())
        km = kmin(len(X), M_JNU, 1)
        jnu_rows.append(dict(file=arq[i], label=lab[i], rpm=r, n_win=len(X), flagged=c,
                             pct=100.0 * c / len(X), kmin=km,
                             decided_faulty=bool(c >= km),
                             correct=bool((c < km) if lab[i] == "normal" else (c >= km))))
JNUDF = pd.DataFrame(jnu_rows)
JNUDF.to_csv(os.path.join(OUT, "jnu_results.csv"), index=False)
print("   gravacoes corretas: %d de %d" % (JNUDF.correct.sum(), len(JNUDF)))
piv = JNUDF.pivot_table(index="label", columns="rpm", values="pct")
print(piv.round(2).to_string())
h = JNUDF[JNUDF.label == "normal"]
print("   metades saudaveis retidas: %s %% (nivel de projeto q = %.0f %%)"
      % (", ".join("%.2f" % v for v in h.pct), 100 * Q))
NUM["jnu"] = dict(correct=int(JNUDF.correct.sum()), n=len(JNUDF),
                  healthy_pct=[float(v) for v in h.pct],
                  by_label={str(k): {str(c): float(v) for c, v in row.items()}
                            for k, row in piv.iterrows()})


# ---------------------------------------------------------------- figuras
def fig_ims(kind, fname):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4))
    for ax, seq in zip(axes, SEQS):
        dd = IMS[seq][1]["day"]
        for b in (1, 2, 3, 4):
            d = IMS[seq][b]
            falhou = b in FAILED[seq]
            y = (pd.Series(100.0 * d["k"] / d["nr"]).rolling(12, center=True, min_periods=1).mean()
                 if kind == "pr" else pd.Series(d["z"]))
            y = y.where(pd.Series(d["mon"]))
            ax.plot(*gapped(dd, y.values), color=BCOL[b], lw=1.1 if falhou else 0.6,
                    alpha=1.0 if falhou else 0.75,
                    label="Bearing %d%s" % (b, " (failed)" if falhou else ""))
        for a, z_ in IMS[seq][1]["bandas"]:
            ax.axvspan(dd[a], dd[min(z_, len(dd) - 1)], color="0.85", lw=0)
        g = IMSDF[(IMSDF.seq == seq) & IMSDF.onset_day.notna()].sort_values("onset_day")
        if len(g):
            ax.axvline(float(g.iloc[0].onset_day), color="crimson", ls="--", lw=0.9)
        if kind == "pr":
            ax.axhline(100.0 * IMS[seq][1]["kmin"] / IMS[seq][1]["nr"], color="0.5", ls="--", lw=0.6)
            ax.set_ylim(-2, 102)
            ax.set_ylabel("$P_r$ (%)")
            prim = g.iloc[0]
            ax.set_title("IMS %s: first alarm on bearing %d%s, %.2f d before the end"
                         % (NOME_IMS[seq], int(prim.bearing),
                            " (failed)" if prim.failed else " (not documented as failed)",
                            float(prim.lead_days)))
        else:
            ax.set_yscale("symlog", linthresh=3)
            ax.axhline(3, color="0.5", ls="--", lw=0.6)
            ax.set_ylabel("$z_r$")
            ax.set_title("IMS %s" % NOME_IMS[seq])
        ax.legend(loc="upper left", ncol=4, frameon=False)
    axes[-1].set_xlabel("Time since the first snapshot (days)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=300)
    plt.close(fig)
    print("   figura %s" % fname)


def fig_pron(conds, ncols, fname, kind="pr"):
    uns = [b for b in PRON if PR[b]["cond"] in conds]
    nrows = int(np.ceil(len(uns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.55 * nrows), squeeze=False)
    for ax, bn in zip(axes.ravel(), uns):
        d = PR[bn]
        x = np.arange(d["n"]) * 10 / 60.0            # minutos
        if kind == "pr":
            y = 100.0 * d["k"] / d["nr"]
            ax.plot(x, y, color="0.75", lw=0.5)
            ax.plot(x, pd.Series(y).rolling(30, center=True, min_periods=1).mean(),
                    color="#1f77b4", lw=1.0)
            ax.axhline(100.0 * d["kmin"] / d["nr"], color="0.5", ls="--", lw=0.6)
            ax.set_ylim(-4, 104)
        ax.axvspan(PR_S0 * 10 / 60.0, (PR_S0 + PR_N) * 10 / 60.0, color="0.85", lw=0)
        if d["onset"] is not None:
            ax.axvline(d["onset"] * 10 / 60.0, color="crimson", ls="--", lw=0.9)
        if d["ntr"] > 0:
            ax.axvline(d["ntr"] * 10 / 60.0, color="0.3", ls="-.", lw=0.7)
        ax.set_title(bn.replace("_", r"\_"), pad=2)
    for ax in axes.ravel()[len(uns):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("min")
    axes[0, 0].set_ylabel("$P_r$ (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=300)
    plt.close(fig)
    print("   figura %s" % fname)


print("\n" + "=" * 78)
print("FIGURAS")
print("=" * 78)
fig_ims("pr", "imsResults_v4.png")
fig_ims("z", "imsZscore_v4.png")
fig_pron((1,), 4, "pronostiaResults_v4.png")
fig_pron((2, 3), 4, "pronostiaResults_c23_v4.png")

# JNU: mapa de calor e caixas
ordem = ["normal", "inner", "ball", "outer"]
rot = {"normal": "N (held out)", "inner": "IR", "ball": "RE", "outer": "OR"}
M = JNUDF.pivot_table(index="label", columns="rpm", values="pct").reindex(ordem)
fig, ax = plt.subplots(figsize=(3.6, 2.4))
im = ax.imshow(M.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(3), ["%d rpm" % c for c in M.columns])
ax.set_yticks(range(len(ordem)), [rot[o] for o in ordem])
for i in range(len(ordem)):
    for j in range(3):
        v = M.values[i, j]
        ax.text(j, i, "%.1f" % v, ha="center", va="center", fontsize=7,
                color="white" if v < 60 else "black")
fig.colorbar(im, ax=ax, label="anomalous windows (%)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "jnuHeatmap_v4.png"), dpi=300)
plt.close(fig)
print("   figura jnuHeatmap_v4.png")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
for ax, r in zip(axes, (600, 800, 1000)):
    i_n = [i for i in range(12) if lab[i] == "normal" and rpm[i] == r][0]
    En = Ej[oj[i_n]:oj[i_n + 1]]
    mu, var = fit(En[a_j:a_j + n_j])
    T = float(np.quantile(scores(En[a_j:a_j + n_j], mu, var), QW))
    dados, nomes = [], []
    for o in ordem:
        i = [k2 for k2 in range(12) if lab[k2] == o and rpm[k2] == r][0]
        X = En[len(En) // 2:] if o == "normal" else Ej[oj[i]:oj[i + 1]]
        dados.append(scores(X, mu, var))
        nomes.append(rot[o].split()[0])
    ax.boxplot(dados, whis=1.5, showfliers=False, labels=nomes, widths=0.6)
    ax.axhline(T, color="crimson", ls="--", lw=0.8)
    ax.set_title("%d rpm" % r)
axes[0].set_ylabel("Mahalanobis score")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "jnuBoxplot_v4.png"), dpi=300)
plt.close(fig)
print("   figura jnuBoxplot_v4.png")

json.dump(NUM, open(os.path.join(OUT, "numeros.json"), "w"), indent=2)
print("\nnumeros em %s" % os.path.relpath(os.path.join(OUT, "numeros.json"), H))


# ------------------------------------------- z-score do PRONOSTIA (condicao 1)
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.2), squeeze=False)
uns = [b for b in PRON if PR[b]["cond"] == 1]
for ax, bn in zip(axes.ravel(), uns):
    E, off, ntr, cond, sub = pron_load(bn)
    mu, var = fit(E[off[PR_S0]:off[PR_S0 + PR_N]])
    s = scores(E, mu, var)
    msc = np.add.reduceat(s, off[:-1]) / np.diff(off)
    ref = msc[PR_S0:PR_S0 + PR_N]
    z = (msc - ref.mean()) / (ref.std(ddof=1) + 1e-12)
    x = np.arange(len(z)) * 10 / 60.0
    ax.plot(x, z, color="#1f77b4", lw=0.8)
    ax.set_yscale("symlog", linthresh=3)
    ax.axhline(3, color="0.5", ls="--", lw=0.6)
    ax.axvspan(PR_S0 * 10 / 60.0, (PR_S0 + PR_N) * 10 / 60.0, color="0.85", lw=0)
    if PR[bn]["onset"] is not None:
        ax.axvline(PR[bn]["onset"] * 10 / 60.0, color="crimson", ls="--", lw=0.9)
    ax.set_title(bn.replace("_", r"\_"), pad=2)
for ax in axes.ravel()[len(uns):]:
    ax.axis("off")
for ax in axes[-1]:
    ax.set_xlabel("min")
axes[0, 0].set_ylabel("$z_r$")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "pronostiaZscore_v4.png"), dpi=300)
plt.close(fig)
print("   figura pronostiaZscore_v4.png")


# ------------------------------------- MAFAULDA: caixas e matriz de confusao
AV = pd.read_csv(os.path.join(H, "rodada7", "mafaulda_avaliacao.csv"))
AV = AV[AV.split != "cal"]
FAM = ["normal", "imbalance", "horizontal-misalignment", "vertical-misalignment",
       "underhang-bearing", "overhang-bearing"]
ROT = {"normal": "Normal", "imbalance": "Imbalance", "horizontal-misalignment": "Horiz. misalign.",
       "vertical-misalignment": "Vert. misalign.", "underhang-bearing": "Underhang bearing",
       "overhang-bearing": "Overhang bearing"}
fig, ax = plt.subplots(figsize=(6.4, 2.8))
dados = [AV[AV.fault_type == f].pct.values for f in FAM]
bp = ax.boxplot(dados, whis=1.5, showfliers=True, labels=[ROT[f] for f in FAM],
                widths=0.6, flierprops=dict(ms=2, mfc="0.4", mec="none"))
ax.axhline(50, color="crimson", ls="--", lw=0.9)
ax.set_ylabel("Anomalous windows (%)")
ax.set_ylim(-4, 104)
plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "boxplotMafaulda_v4.png"), dpi=300)
plt.close(fig)
print("   figura boxplotMafaulda_v4.png")

y = (AV.fault_type != "normal").values
yh = (AV.pct >= 50.0).values
cm = np.array([[int((~y & ~yh).sum()), int((~y & yh).sum())],
               [int((y & ~yh).sum()), int((y & yh).sum())]])
fig, ax = plt.subplots(figsize=(3.0, 2.6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Normal", "Fault"])
ax.set_yticks([0, 1], ["Normal", "Fault"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(2):
    for j in range(2):
        ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "confusioMatrix_v4.png"), dpi=300)
plt.close(fig)
print("   figura confusioMatrix_v4.png | matriz %s" % cm.tolist())
NUM["mafaulda_cm"] = cm.tolist()
json.dump(NUM, open(os.path.join(OUT, "numeros.json"), "w"), indent=2)
