"""
Analises de sensibilidade do artigo -- PROTOCOLO FINAL (rodada 7).

Refaz as tres tabelas que ainda traziam numeros da rodada 5:

  A  tab:ims_sweep            inicio x duracao da referencia saudavel
  B  tab:persistence          plato de K e a menor referencia por K
  C  tab:commissioning_folds  a regra calibrada em outros grupos de unidades

Tudo com: embeddings da rodada 7 (rerun7), vetor de variancias por coordenada,
re-comissionamento apos parada documentada, q = 0,01, alfa = 0,05, M = 10^4.
A duracao da referencia de re-comissionamento acompanha a duracao n que esta
sendo avaliada, como diz a Secao 2.3.4.

Saidas: final_v4/sens_*.csv e sens_resumo.txt

    cd code && python analise_sensibilidade_v4.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
__FONTE = os.path.join(H, "analise_final_v4.py")
exec(compile(io.open(__FONTE, encoding="utf-8").read().split('print("=" * 78)')[0],
             "analise_final_v4.py", "exec"))
H = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.join(H, "rerun7")
OUT = os.path.join(H, "final_v4")

S0_H = (0, 6, 12, 24, 36, 48, 72, 96)
N_H = (1 / 6, 1, 3, 6, 12, 18, 24, 36, 48, 72, 96)
KS = tuple(range(1, 31))
LOG = []


def diz(*a):
    t = " ".join(str(x) for x in a)
    print(t, flush=True)
    LOG.append(t)


# Cache em dois niveis. O caro e pontuar a matriz inteira de embeddings, e a
# mesma referencia reaparece em muitos pontos da grade: as referencias de
# re-comissionamento dependem so de n, nao de s0. Guardamos a mascara de
# snapshot anomalo por referencia. Em float32, que basta para decidir o alarme.
CACHE, FLAGS = {}, {}


def dados(seq, b):
    if (seq, b) not in CACHE:
        E, off, day = ims_load(seq, b)
        CACHE[(seq, b)] = (np.ascontiguousarray(E, dtype=np.float32), off, day,
                           ims_db(seq, b), kmin(int(np.diff(off).max()), M_IMS, 4),
                           list(np.where(np.diff(day) * 24 >= GAP_H)[0] + 1))
    return CACHE[(seq, b)]


def flags(seq, b, r0, rn):
    """Mascara de snapshot anomalo para a referencia (r0, rn), com cache."""
    ch = (seq, b, r0, rn)
    if ch not in FLAGS:
        E, off, day, _, km, _ = dados(seq, b)
        Ref = E[off[r0]:off[r0 + rn]]
        mu = Ref.mean(axis=0)
        var = Ref.var(axis=0, ddof=1) + np.float32(EPS)
        D = E - mu
        s = np.sqrt(np.einsum("ij,ij->i", D * D, np.broadcast_to(1.0 / var, D.shape)))
        T = float(np.quantile(s[off[r0]:off[r0 + rn]], QW))
        kk = np.add.reduceat((s > T).astype(np.int64), off[:-1])
        FLAGS[ch] = kk >= km
    return FLAGS[ch]


def roda(seq, b, s0, n, K_=K, delta=DELTA_DB):
    """Detector completo com re-comissionamento. Devolve (dia do alarme, indice)."""
    E, off, day, db0, km, paradas = dados(seq, b)
    if s0 + n > len(day):
        return None
    # O aceite compara o RMS depois da parada com o da referencia de comissionamento
    # ORIGINAL do candidato (Sec. 2.3.5), nao com a de 36 h + 18 h que ims_db usa
    # de base: refaz a base de db0 no RMS medio linear de s0 .. s0 + n.
    db0 = db0 - 20 * np.log10(np.mean(10.0 ** (db0[s0:s0 + n] / 20.0)))
    refs = [(s0, n, True)]
    for a in paradas:
        cabe = a + n + K_ <= len(day)
        refs.append((a, n, bool(cabe and abs(float(db0[a:a + n].mean())) <= delta)))
    cur, alarme = (s0, n), None
    for i, (r0, rn, ok) in enumerate(refs):
        if ok:
            cur = (r0, rn)
        ini = max(cur[0] + cur[1], paradas[i - 1] if i else 0)
        fim = paradas[i] if i < len(paradas) else len(day)
        if ini >= fim or cur[0] + cur[1] > len(day):
            continue
        j = onset(flags(seq, b, cur[0], cur[1])[ini:fim], 0, K_)
        if j is not None and alarme is None:
            alarme = (day[ini + j], ini + j)
    return alarme


def primeiro(seq, s0, n, K_=K):
    """Ordena os quatro rolamentos pelo alarme; devolve (rol, dia, lead, saudaveis antes)."""
    day = dados(seq, 1)[2]
    fim = day[-1]
    al = {}
    for b in (1, 2, 3, 4):
        r = roda(seq, b, s0, n, K_)
        if r is not None:
            al[b] = r[0]
    if not al:
        return None
    ordem = sorted(al, key=al.get)
    falhas = [b for b in ordem if b in FAILED[seq]]
    tf = al[falhas[0]] if falhas else np.inf
    fa = [b for b in ordem if b not in FAILED[seq] and al[b] < tf]
    return dict(bearing=ordem[0], day=al[ordem[0]], lead=fim - al[ordem[0]],
                failed=ordem[0] in FAILED[seq], healthy_before=fa, all=al, end=fim)


# ===================================================================== A
diz("=" * 78)
diz("A -- sensibilidade ao inicio e a duracao da referencia (tab:ims_sweep)")
diz("=" * 78)
linhas = []
for s0h in S0_H:
    for nh in N_H:
        s0, n = int(s0h * 6), max(1, int(round(nh * 6)))
        cel = {}
        for seq in SEQS:
            day = dados(seq, 1)[2]
            if s0 + n + K + 5 > len(day):
                cel[seq] = None
                continue
            cel[seq] = primeiro(seq, s0, n)
        linhas.append(dict(s0_h=s0h, n_h=round(nh, 3), n_windows=n * 39,
                           **{seq: (None if cel[seq] is None else
                                    "%d%s (%.2f)" % (cel[seq]["bearing"],
                                                     "" if cel[seq]["failed"] else "*",
                                                     cel[seq]["lead"])) for seq in SEQS},
                           **{seq + "_fa": (None if cel[seq] is None else
                                            ",".join(map(str, cel[seq]["healthy_before"])))
                              for seq in SEQS}))
        diz("   s0 %3d h  n %6.2f h (%5d janelas) | %s" % (
            s0h, nh, n * 39,
            " | ".join("%-12s" % (linhas[-1][seq] or "curto") for seq in SEQS)))
SWEEP = pd.DataFrame(linhas)
SWEEP.to_csv(os.path.join(OUT, "sens_ims_sweep.csv"), index=False)
io.open(os.path.join(OUT, "sens_resumo.txt"), "w", encoding="utf-8").write("\n".join(LOG))


# ===================================================================== B
diz("")
diz("=" * 78)
diz("B -- persistencia K (tab:persistence e o plato)")
diz("=" * 78)
diz("   No comissionamento adotado (36 h + 18 h): primeiro alarme por sequencia")
diz("   %-4s %-26s %-26s %-26s" % ("K", *SEQS))
linhas_k = []
for Kv in KS:
    cel = {}
    for seq in SEQS:
        r = primeiro(seq, 216, 108, Kv)
        cel[seq] = r
    diz("   %-4d %s" % (Kv, " ".join(
        "%-26s" % ("rol %d%s d%.2f (%s)" % (cel[s]["bearing"], "" if cel[s]["failed"] else "*",
                                            cel[s]["day"],
                                            ",".join(map(str, cel[s]["healthy_before"])) or "-")
                   if cel[s] else "sem alarme") for s in SEQS)))
    linhas_k.append(dict(K=Kv, **{s: (None if not cel[s] else cel[s]["bearing"]) for s in SEQS},
                         **{s + "_day": (None if not cel[s] else cel[s]["day"]) for s in SEQS},
                         **{s + "_lead": (None if not cel[s] else cel[s]["lead"]) for s in SEQS},
                         **{s + "_fa": (None if not cel[s] else
                                        ",".join(map(str, cel[s]["healthy_before"]))) for s in SEQS}))
pd.DataFrame(linhas_k).to_csv(os.path.join(OUT, "sens_K.csv"), index=False)

diz("")
diz("   Menor duracao (inicio fixo em 36 h) sem alarme saudavel antes da falha")
diz("   em 1st_test e 2nd_test, por K. O 4th_test e listado a parte porque o")
diz("   rolamento 4 alarma em todos os pontos da grade.")
diz("   %-6s %-10s %-34s %s" % ("K", "duracao", "antecedencias (1st, 2nd)", "4th: 1o alarme"))
linhas_kn = []
for Kv in KS:
    achou = None
    for nh in N_H:
        n = max(1, int(round(nh * 6)))
        if n * DUR_IMS < PISO_SINAL:
            continue
        r1, r2 = primeiro("1st_test", 216, n, Kv), primeiro("2nd_test", 216, n, Kv)
        if r1 and r2 and r1["failed"] and r2["failed"] and not r1["healthy_before"] \
                and not r2["healthy_before"]:
            achou = (nh, n, r1, r2)
            break
    if achou is None:
        diz("   %-6d %-10s %-34s %s" % (Kv, "nenhuma", "--", "--"))
        linhas_kn.append(dict(K=Kv, n_h=None))
        continue
    nh, n, r1, r2 = achou
    r4 = primeiro("4th_test", 216, n, Kv)
    diz("   %-6d %-10s %-34s %s" % (
        Kv, "%g h" % nh, "%.2f, %.2f d" % (r1["lead"], r2["lead"]),
        "rol %d%s d%.2f" % (r4["bearing"], "" if r4["failed"] else "*", r4["day"]) if r4 else "-"))
    linhas_kn.append(dict(K=Kv, n_h=nh, lead_1st=r1["lead"], lead_2nd=r2["lead"],
                          b_4th=None if not r4 else r4["bearing"],
                          day_4th=None if not r4 else r4["day"],
                          lead_4th=None if not r4 else r4["lead"]))
pd.DataFrame(linhas_kn).to_csv(os.path.join(OUT, "sens_K_menor_ref.csv"), index=False)
io.open(os.path.join(OUT, "sens_resumo.txt"), "w", encoding="utf-8").write("\n".join(LOG))
diz("")
diz("resumo em final_v4/sens_resumo.txt")


# ===================================================================== C
# A regra de comissionamento rodada em grupos de calibracao diferentes.
# Passos da Secao 2.3.3: candidatos na grade; minimo de 1/q janelas; falso
# alarme (a) alarme em dado documentado como saudavel, (b) alarme dentro das
# V aquisicoes seguintes a referencia; aceitacao com tolerancia floor(0.1 U);
# estabilidade nos vizinhos; escolha pela menor duracao, empate pelo inicio.

S0_P, N_P = (0, 3, 6, 18, 30, 60), (1, 3, 6, 12, 18, 30, 60, 100)
S0_J, N_J = (0, 2, 5, 10), (0.25, 0.5, 1, 2, 3, 5, 10, 15)
WPS_J = 2931 / 30.0


def escolhe(aceito, s0s, ns):
    """Estabilidade nos vizinhos, depois menor duracao e menor inicio."""
    melhor = None
    for i, s0 in enumerate(s0s):
        for j, n in enumerate(ns):
            if not aceito.get((s0, n), False):
                continue
            viz = [k for k in (((s0s[i + 1], n) if i + 1 < len(s0s) else None),
                               ((s0, ns[j + 1]) if j + 1 < len(ns) else None))
                   if k is not None and k in aceito]
            if not all(aceito[k] for k in viz):
                continue
            if melhor is None or (n, s0) < melhor[0]:
                melhor = ((n, s0), (s0, n))
    return None if melhor is None else melhor[1]


# ---------------------------------------------------------------- IMS
def ims_falso_alarme(seq, s0, n):
    """(a) saudavel antes da falha; (b) alarme dentro de V apos a referencia."""
    r = primeiro(seq, s0, n)
    if r is None:
        return False
    if r["healthy_before"]:
        return True
    return any(i - (s0 + n) < VA_IMS for i in
               [int(np.searchsorted(dados(seq, 1)[2], d)) for d in r["all"].values()]
               if i >= s0 + n)


diz("")
diz("=" * 78)
diz("C -- a regra calibrada em grupos diferentes (tab:commissioning_folds)")
diz("=" * 78)
diz("")
diz("IMS, tolerancia floor(0.1 U) = 0 para qualquer U <= 9")
GRUPOS_IMS = [("1st + 2nd (declarado)", ("1st_test", "2nd_test")),
              ("as tres sequencias", SEQS),
              ("so 1st_test", ("1st_test",)),
              ("so 2nd_test", ("2nd_test",)),
              ("so 4th_test", ("4th_test",))]
FA_IMS = {}
for s0h in S0_H:
    for nh in N_H:
        s0, n = int(s0h * 6), max(1, int(round(nh * 6)))
        if n * DUR_IMS < PISO_SINAL:
            continue
        for seq in SEQS:
            if s0 + n + K + 5 <= len(dados(seq, 1)[2]):
                FA_IMS[(seq, s0h, round(nh, 3))] = ims_falso_alarme(seq, s0, n)
linhas_c = []
for nome, grupo in GRUPOS_IMS:
    aceito = {}
    for s0h in S0_H:
        for nh in N_H:
            ch = [FA_IMS.get((seq, s0h, round(nh, 3))) for seq in grupo]
            if any(c is None for c in ch):
                continue
            aceito[(s0h, round(nh, 3))] = sum(ch) <= int(0.1 * len(grupo))
    sel = escolhe(aceito, S0_H, [round(x, 3) for x in N_H])
    n_ok = sum(aceito.values())
    if sel is None:
        diz("   %-24s %4d de %3d pontos aceitos | nenhuma referencia selecionada"
            % (nome, n_ok, len(aceito)))
        linhas_c.append(dict(base="IMS", grupo=nome, aceitos=n_ok, total=len(aceito), sel=None))
        continue
    s0, n = int(sel[0] * 6), max(1, int(round(sel[1] * 6)))
    fora = [seq for seq in SEQS if seq not in grupo]
    res = []
    for seq in fora:
        r = primeiro(seq, s0, n)
        res.append("%s: rol %d%s %.2f d" % (seq.replace("_test", ""), r["bearing"],
                                            "" if r["failed"] else "*", r["lead"]))
    diz("   %-24s %4d de %3d pontos | selecionado %g h + %g h | fora do grupo: %s"
        % (nome, n_ok, len(aceito), sel[0], sel[1], "; ".join(res) or "nada"))
    linhas_c.append(dict(base="IMS", grupo=nome, aceitos=n_ok, total=len(aceito),
                         sel="%g h + %g h" % sel, fora="; ".join(res)))
pd.DataFrame(linhas_c).to_csv(os.path.join(OUT, "sens_folds.csv"), index=False)
io.open(os.path.join(OUT, "sens_resumo.txt"), "w", encoding="utf-8").write("\n".join(LOG))


# --------------------------------------------------------- PRONOSTIA e JNU
def pron_res(bn, s0, n):
    z = np.load(os.path.join(R7, "A", "pronostia", "pronostia_rep_win_%s_emb.npz" % bn))
    E, off = z["E"].astype(np.float32), z["offsets"]
    ns = len(off) - 1
    if s0 + n + K > ns:
        return None
    km = kmin(int(np.diff(off).max()), M_PRON, 1)
    Ref = E[off[s0]:off[s0 + n]]
    mu, var = Ref.mean(axis=0), Ref.var(axis=0, ddof=1) + np.float32(EPS)
    D = E - mu
    s = np.sqrt(np.einsum("ij,ij->i", D * D, np.broadcast_to(1.0 / var, D.shape)))
    T = float(np.quantile(s[off[s0]:off[s0 + n]], QW))
    k = np.add.reduceat((s > T).astype(np.int64), off[:-1])
    i = onset(k >= km, s0 + n)
    return dict(n=ns, onset=i, rem=None if i is None else 100.0 * (ns - i) / ns,
                lead_h=None if i is None else (ns - i) * 10 / 3600.0,
                val=bool(i is not None and i - (s0 + n) < VA_PRON), cond=int(bn[7]))


JN = np.load(os.path.join(R7, "A", "jnu", "jnu_rep_win_emb.npz"))
EJ, OJ = JN["E"].astype(np.float32), JN["offsets"]
LABJ, RPMJ = list(JN["labels"]), list(JN["rpm"])


def jnu_res(rpm_, s0_s, n_s):
    a, n = int(round(s0_s * WPS_J)), int(round(n_s * WPS_J))
    i_n = [i for i in range(12) if LABJ[i] == "normal" and RPMJ[i] == rpm_][0]
    En = EJ[OJ[i_n]:OJ[i_n + 1]]
    meio = len(En) // 2
    if a + n > meio:
        return None
    Ref = En[a:a + n]
    mu, var = Ref.mean(axis=0), Ref.var(axis=0, ddof=1) + np.float32(EPS)

    def sc(X):
        D = X - mu
        return np.sqrt(np.einsum("ij,ij->i", D * D, np.broadcast_to(1.0 / var, D.shape)))
    T = float(np.quantile(sc(Ref), QW))
    out = {}
    X = En[meio:]
    out["normal"] = (int((sc(X) > T).sum()), len(X), kmin(len(X), M_JNU, 1))
    for o in ("inner", "outer", "ball"):
        i = [k2 for k2 in range(12) if LABJ[k2] == o and RPMJ[k2] == rpm_][0]
        X = EJ[OJ[i]:OJ[i + 1]]
        out[o] = (int((sc(X) > T).sum()), len(X), kmin(len(X), M_JNU, 1))
    return out


diz("")
diz("PRONOSTIA, criterio (b) apenas (nao ha rolamento documentado como saudavel)")
FA_P = {}
for s0 in S0_P:
    for n in N_P:
        if n * DUR_PRON < PISO_SINAL:
            continue
        for bn in PRON:
            r = pron_res(bn, s0, n)
            FA_P[(bn, s0, n)] = None if r is None else r["val"]
for nome, conds in (("condicao 1 (declarada)", (1,)), ("condicao 2", (2,)),
                    ("condicao 3", (3,)), ("todas as condicoes", (1, 2, 3))):
    uns = [b for b in PRON if int(b[7]) in conds]
    aceito = {}
    for s0 in S0_P:
        for n in N_P:
            ch = [FA_P.get((b, s0, n)) for b in uns]
            if any(c is None for c in ch):
                continue
            aceito[(s0, n)] = sum(ch) <= int(0.1 * len(uns))
    sel = escolhe(aceito, S0_P, N_P)
    if sel is None:
        diz("   %-24s %3d de %2d pontos | nenhuma referencia" % (nome, sum(aceito.values()), len(aceito)))
        continue
    fora = [b for b in PRON if b not in uns]
    rs = [pron_res(b, *sel) for b in fora]
    ok = [r for r in rs if r and r["onset"] is not None]
    diz("   %-24s %3d de %2d pontos | selecionado %d + %d aquisicoes (%.0f + %.0f s) | "
        "fora do grupo: %d de %d alarmam, restante mediano %.1f %%"
        % (nome, sum(aceito.values()), len(aceito), sel[0], sel[1], sel[0] * 10, sel[1] * 10,
           len(ok), len(fora), np.median([r["rem"] for r in ok]) if ok else float("nan")))

diz("")
diz("JNU, criterio (a): a metade retida da gravacao normal nao pode ser dada como falha")
FA_J = {}
for s0 in S0_J:
    for n in N_J:
        if n < PISO_SINAL:
            continue
        for r_ in (600, 800, 1000):
            out = jnu_res(r_, s0, n)
            FA_J[(r_, s0, n)] = None if out is None else (out["normal"][0] >= out["normal"][2])
for nome, rr in (("600 rpm (declarada)", (600,)), ("800 rpm", (800,)),
                 ("1000 rpm", (1000,)), ("as tres rotacoes", (600, 800, 1000))):
    aceito = {}
    for s0 in S0_J:
        for n in N_J:
            ch = [FA_J.get((r_, s0, n)) for r_ in rr]
            if any(c is None for c in ch):
                continue
            aceito[(s0, n)] = sum(ch) <= int(0.1 * len(rr))
    sel = escolhe(aceito, S0_J, N_J)
    if sel is None:
        diz("   %-22s %2d de %2d pontos | nenhuma referencia" % (nome, sum(aceito.values()), len(aceito)))
        continue
    fora = [r_ for r_ in (600, 800, 1000) if r_ not in rr]
    txt = []
    for r_ in fora:
        o = jnu_res(r_, *sel)
        cert = sum(1 for k2, v in o.items()
                   if ((v[0] < v[2]) if k2 == "normal" else (v[0] >= v[2])))
        txt.append("%d rpm %d/4 corretas (saudavel %.2f %%)"
                   % (r_, cert, 100.0 * o["normal"][0] / o["normal"][1]))
    diz("   %-22s %2d de %2d pontos | selecionado %g + %g s | %s"
        % (nome, sum(aceito.values()), len(aceito), sel[0], sel[1], "; ".join(txt) or "nada fora"))

io.open(os.path.join(OUT, "sens_resumo.txt"), "w", encoding="utf-8").write("\n".join(LOG))
diz("")
diz("fim")
