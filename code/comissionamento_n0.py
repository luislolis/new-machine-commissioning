# -*- coding: utf-8 -*-
"""
Comissionamento de nivel 0: a referencia cresce enquanto e homogenea, recua e trava.

Declaracao: resultados-nivel0.md, Secao 1 (escrita antes de rodar). Nenhum rotulo de falha entra
na regra: FAILED so aparece em avalia_ims, depois da trava.

    cd code && python comissionamento_n0.py conferir          verificacao previa (r9)
    cd code && python comissionamento_n0.py r9 rnd0 ... resumo

Variantes: "var" (regra principal: trava no primeiro intervalo heterogeneo, recuando um bloco),
"teto" (trava a 1000 janelas homogeneas; heterogeneidade antes disso recomeca) e "prog" (a
principal, com as series em ordem e heranca do comissionamento das anteriores).
"""
import gc
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "final_n0")
DADOS = os.environ.get("MISAEL_DADOS") or H     # data next to the scripts unless MISAEL_DADOS says otherwise
K8 = 7
TETO_JAN = 1000
VARIANTES = ("var", "teto", "prog", "viz")             # viz: segunda declaracao (resultados-nivel0.md, Sec. 3)
REPRES = ("rnd0", "rnd1", "specn", "spec", "feat", "featn", "ce", "zer")


def troca(src, a, b):
    assert src.count(a) == 1, a
    return src.replace(a, b)


R12 = ("r9_12", "r12", "r9nc", "rnd0nc", "r12s", "r12l", "r12g", "r12p")   # rerun_extract12; r9_12 = rerun12/r9


def pasta(tag):
    if tag == "r9":
        return os.path.join(H, "rerun9")
    if tag in R12:
        return os.path.join(DADOS, "rerun12", "r9" if tag == "r9_12" else tag)
    if tag in REPRES or tag == "r9rms":                 # r9rms: deriva_r9rms.py (r9 + log RMS por canal)
        return os.path.join(DADOS, "rerun11", tag)
    return os.path.join(DADOS, "rerun10", tag)


def ambiente(rerun):
    """O mesmo SV do analise_final_v8.py (linhas 67-78), com a pasta de embeddings trocada."""
    v4 = io.open(os.path.join(H, "analise_final_v4.py"), encoding="utf-8").read()
    v4k = troca(v4, "Q, ALPHA, K, EPS = 0.01, 0.05, 10, 1e-12", "Q, ALPHA, K, EPS = 0.01, 0.05, %d, 1e-12" % K8)
    sens = io.open(os.path.join(H, "analise_sensibilidade_v4.py"), encoding="utf-8").read()
    cab = sens[:sens.index("# " + "=" * 69 + " A")]
    cab = troca(cab, 'io.open(__FONTE, encoding="utf-8").read()', "__V4K")
    cab = troca(cab, 'R7 = os.path.join(H, "rerun7")', "R7 = %r" % rerun)
    d1 = sens[sens.index("S0_P, N_P = "):sens.index('diz("")\ndiz("=" * 78)\ndiz("C --')]
    d2 = sens[sens.index("# --------------------------------------------------------- PRONOSTIA e JNU"):sens.index("FA_P = {}")]
    sv = {"__name__": "n0", "__file__": os.path.join(H, "analise_sensibilidade_v4.py"), "__V4K": v4k}
    cwd = os.getcwd()
    os.chdir(H)
    try:
        exec(compile(cab + "\n" + d1 + "\n" + d2, "analise_sensibilidade_v4.py[n0]", "exec"), sv)
    finally:
        os.chdir(cwd)
    assert sv["K"] == K8 and sv["R7"] == rerun
    return sv


# ------------------------------------------------------------------ pontuacao
def pontua(E, mu, var):
    D = E - mu
    return np.sqrt(np.einsum("ij,ij->i", D * D, np.broadcast_to(1.0 / var, D.shape)))


def modelo(E, a, b):
    """mu, var e T na faixa de janelas [a, b)."""
    R = E[a:b]
    mu = R.mean(axis=0)
    var = R.var(axis=0, ddof=1) + np.float32(1e-12)
    T = float(np.quantile(pontua(R, mu, var), 0.99))
    return mu, var, T


def alarme_em(E, off, km, a0, a1, b1, onset):
    """Detector comissionado nas aquisicoes [a0, a1) e rodado em [a1, b1): ha alarme (K)?"""
    mu, var, T = modelo(E, off[a0], off[a1])
    s = pontua(E[off[a1]:off[b1]], mu, var)
    k = np.add.reduceat((s > T).astype(np.int64), off[a1:b1] - off[a1])
    return onset(k >= km, 0, K8) is not None


def alarme_na_segunda_metade(E, off, km, s0, n, onset):
    """Detector comissionado em [s0, s0 + n//2) e rodado em [s0 + n//2, s0 + n): ha alarme (K)?"""
    return alarme_em(E, off, km, s0, s0 + n // 2, s0 + n, onset)


def alarme_vizinhas(E, off, km, s0, n, w, onset):
    """Variante viz: janelas vizinhas de tamanho w no fim do intervalo (so quando n >= 2w)."""
    if n < 2 * w:
        return False
    return alarme_em(E, off, km, s0 + n - 2 * w, s0 + n - w, s0 + n, onset)


def w_viz(bloco):
    return bloco * int(np.ceil(K8 / bloco))


# ------------------------------------------------------------------ a regra N0
def cresce(heterogeneo, total, bloco, paradas, minimo_trava=0, s0_ini=0, teto=None, trava_no_fim=False):
    """Cresce, recua e trava. heterogeneo(s0, n) -> bool. Devolve dict com a trava ou None.

    teto=None: regra principal (trava no primeiro heterogeneo, recuando um bloco; recuo menor que
    max(2 blocos, minimo_trava) recomeca em s0 + n). teto=t: trava quando n >= t homogeneo; qualquer
    heterogeneidade antes recomeca em s0 + n. trava_no_fim (so no JNU, cuja janela de comissionamento
    e a primeira metade da gravacao por desenho): no fim dessa janela, trava no ultimo homogeneo."""
    s0, n, recomecos = s0_ini, 2 * bloco, []
    while True:
        prox = [p for p in paradas if s0 < p]
        lim = min(prox[0] if prox else total, total)
        if s0 + n > lim:
            if prox and prox[0] < total:                       # parada dentro do intervalo
                recomecos.append(("parada", int(s0), int(n)))
                s0, n = prox[0], 2 * bloco
                continue
            if trava_no_fim and n - bloco >= 2 * bloco:
                return dict(trava=(int(s0), int(n - bloco)), recomecos=recomecos, motivo="fim da janela, homogeneo")
            return dict(trava=None, recomecos=recomecos, motivo="fim dos dados sem travar")
        if heterogeneo(s0, n):
            r = n - bloco
            if teto is None and r >= max(2 * bloco, minimo_trava):
                return dict(trava=(int(s0), int(r)), recomecos=recomecos, motivo="variabilidade em n=%d" % n)
            recomecos.append(("heterogeneo", int(s0), int(n)))
            s0, n = s0 + n, 2 * bloco
            continue
        if teto is not None and n >= teto:
            return dict(trava=(int(s0), int(n)), recomecos=recomecos, motivo="teto homogeneo")
        n += bloco


# ------------------------------------------------------------------ IMS
def ims_maquina(SV, seq, variante, herda):
    dados, onset = SV["dados"], SV["onset"]
    U = {b: dados(seq, b) for b in (1, 2, 3, 4)}
    E1, off1, day = U[1][0], U[1][1], U[1][2]
    paradas = list(U[1][5])
    total = len(day)
    nr = int(np.diff(off1).max())
    bloco = 3

    w = w_viz(bloco)

    def het(s0, n):
        if any(alarme_na_segunda_metade(U[b][0], U[b][1], U[b][4], s0, n, onset) for b in U):
            return True
        return variante == "viz" and any(alarme_vizinhas(U[b][0], U[b][1], U[b][4], s0, n, w, onset) for b in U)

    teto = None
    if variante == "teto":
        teto = bloco * int(np.ceil(TETO_JAN / (nr * bloco)))
    r = cresce(het, total, bloco, paradas, minimo_trava=herda.get("n", 0) if variante == "prog" else 0,
               s0_ini=herda.get("s0", 0) if variante == "prog" else 0, teto=teto)
    r["teto"] = teto
    if r["trava"]:
        s0, n = r["trava"]
        r["horas_aquisicao"] = (s0 * 10 / 60.0, n * 10 / 60.0)
        r["dias_calendario"] = (float(day[s0]), float(day[s0 + n - 1]))
    return r


def roda0(SV, seq, b, s0, n):
    """roda() de analise_sensibilidade_v4.py, so com as paradas posteriores a s0 + n."""
    E, off, day, db0, km, paradas = SV["dados"](seq, b)
    paradas = [a for a in paradas if a >= s0 + n]
    db0 = db0 - 20 * np.log10(np.mean(10.0 ** (db0[s0:s0 + n] / 20.0)))
    refs = [(s0, n, True)]
    for a in paradas:
        cabe = a + n + K8 <= len(day)
        refs.append((a, n, bool(cabe and abs(float(db0[a:a + n].mean())) <= SV["DELTA_DB"])))
    cur = (s0, n)
    for i, (r0, rn, ok) in enumerate(refs):
        if ok:
            cur = (r0, rn)
        ini = max(cur[0] + cur[1], paradas[i - 1] if i else 0)
        fim = paradas[i] if i < len(paradas) else len(day)
        if ini >= fim or cur[0] + cur[1] > len(day):
            continue
        j = SV["onset"](SV["flags"](seq, b, cur[0], cur[1])[ini:fim], 0, K8)
        if j is not None:
            return float(day[ini + j]), int(ini + j)
    return None


def avalia_ims(SV, seq, ref, roda=None):
    """Desfecho: aqui, e so aqui, entram os rolamentos que falharam."""
    if ref is None:
        return dict(sem_trava=True)
    day = SV["dados"](seq, 1)[2]
    fim = float(day[-1])
    al = {}
    for b in (1, 2, 3, 4):
        x = (roda or (lambda s, bb, a, m: roda0(SV, s, bb, a, m)))(seq, b, *ref)
        if x is not None:
            al[b] = x
    if not al:
        return dict(primeiro=None)
    p = min(al, key=lambda b: al[b][0])
    fal = [b for b in SV["FAILED"][seq] if b in al]
    tf = min(al[b][0] for b in fal) if fal else np.inf
    return dict(primeiro=int(p), certo=bool(p in SV["FAILED"][seq]), lead_d=round(fim - al[p][0], 2),
                onset=int(al[p][1]), saudaveis_antes=sorted(int(b) for b in al
                                                            if b not in SV["FAILED"][seq] and al[b][0] < tf),
                alarmes={int(b): [int(v[1]), round(fim - v[0], 2)] for b, v in al.items()})


# ------------------------------------------------------------------ PRONOSTIA
def pron_dados(SV, bn):
    z = np.load(os.path.join(SV["R7"], "A", "pronostia", "pronostia_rep_win_%s_emb.npz" % bn))
    E, off = z["E"].astype(np.float32), z["offsets"]
    return E, off, SV["kmin"](int(np.diff(off).max()), SV["M_PRON"], 1)


def pron_maquina(SV, bn, variante, herda):
    E, off, km = pron_dados(SV, bn)
    total, bloco = len(off) - 1, 30

    w = w_viz(bloco)

    def het(s0, n):
        if alarme_na_segunda_metade(E, off, km, s0, n, SV["onset"]):
            return True
        return variante == "viz" and alarme_vizinhas(E, off, km, s0, n, w, SV["onset"])

    teto = bloco * int(np.ceil(TETO_JAN / (4 * bloco))) if variante == "teto" else None
    r = cresce(het, total, bloco, [], minimo_trava=herda.get("n", 0) if variante == "prog" else 0,
               s0_ini=herda.get("s0", 0) if variante == "prog" else 0, teto=teto)
    r["teto"] = teto
    if r["trava"]:
        s0, n = r["trava"]
        x = SV["pron_res"](bn, s0, n)
        r["desfecho"] = None if x is None else dict(
            onset=x["onset"], rem=None if x["rem"] is None else round(x["rem"], 2),
            lead_h=None if x["lead_h"] is None else round(x["lead_h"], 3), n=x["n"])
    return r


# ------------------------------------------------------------------ JNU
def jnu_maquina(SV, rpm, variante):
    EJ, OJ, LAB, RPM = SV["EJ"], SV["OJ"], SV["LABJ"], SV["RPMJ"]
    i_n = [i for i in range(12) if LAB[i] == "normal" and RPM[i] == rpm][0]
    En = EJ[OJ[i_n]:OJ[i_n + 1]]
    meio, bloco = len(En) // 2, int(round(SV["PISO_SINAL"] * SV["WPS_J"]))

    def entre(a0, a1, b1):
        mu, var, T = modelo(En, a0, a1)
        k = int((pontua(En[a1:b1], mu, var) > T).sum())
        return k >= SV["kmin"](b1 - a1, SV["M_JNU"], 1)

    def het(s0, n):
        if entre(s0, s0 + n // 2, s0 + n):
            return True
        return variante == "viz" and n >= 2 * bloco and entre(s0 + n - 2 * bloco, s0 + n - bloco, s0 + n)

    teto = bloco * int(np.ceil(TETO_JAN / bloco)) if variante == "teto" else None
    r = cresce(het, meio, bloco, [], teto=teto, trava_no_fim=True)
    r["teto"] = teto
    if r["trava"]:
        r["desfecho"] = jnu_desfecho(SV, rpm, *r["trava"])
    return r


def jnu_desfecho(SV, rpm, a, n):
    EJ, OJ, LAB, RPM = SV["EJ"], SV["OJ"], SV["LABJ"], SV["RPMJ"]
    i_n = [i for i in range(12) if LAB[i] == "normal" and RPM[i] == rpm][0]
    En = EJ[OJ[i_n]:OJ[i_n + 1]]
    meio = len(En) // 2
    mu, var, T = modelo(En, a, a + n)
    out = {}
    for nome, X in [("normal", En[meio:])] + [(o, EJ[OJ[i]:OJ[i + 1]]) for o in ("inner", "outer", "ball")
                                             for i in range(12) if LAB[i] == o and RPM[i] == rpm]:
        k = int((pontua(X, mu, var) > T).sum())
        km = SV["kmin"](len(X), SV["M_JNU"], 1)
        out[nome] = dict(k=k, n=len(X), kmin=km, falha=bool(k >= km), certo=bool((k >= km) == (nome != "normal")))
    out["corretas"] = sum(v["certo"] for v in out.values())
    return out


# ------------------------------------------------------------------ uma etiqueta
def roda_tag(tag, variantes=VARIANTES):
    rerun = pasta(tag)
    if not os.path.isdir(rerun):
        print("%s: %s nao existe, pulando" % (tag, rerun), flush=True)
        return None
    SV = ambiente(rerun)
    p = os.path.join(OUT, tag, "resultado.json")
    res = json.load(open(p)) if os.path.exists(p) else dict(tag=tag, rerun=rerun)
    t0 = time.time()
    for var in variantes:
        R = res[var] = dict(ims={}, pron={}, jnu={})
        herda = {}
        for seq in SV["SEQS"]:                                   # ja em ordem de data
            m = ims_maquina(SV, seq, var, herda)
            m["desfecho"] = avalia_ims(SV, seq, m["trava"])
            R["ims"][seq] = m
            if var == "prog" and m["trava"]:
                herda = dict(s0=max(herda.get("s0", 0), m["trava"][0]), n=max(herda.get("n", 0), m["trava"][1]))
        herda = {}
        for bn in SV["PRON"]:                                    # condicao 1, 2, 3 e indice
            m = pron_maquina(SV, bn, var, herda)
            R["pron"][bn] = m
            if var == "prog" and m["trava"]:
                herda = dict(s0=max(herda.get("s0", 0), m["trava"][0]), n=max(herda.get("n", 0), m["trava"][1]))
        for rpm in (600, 800, 1000):
            R["jnu"][rpm] = jnu_maquina(SV, rpm, "var" if var == "prog" else var)
        print("%s %s pronto (%.0f s)" % (tag, var, time.time() - t0), flush=True)
    SV["CACHE"].clear()
    SV["FLAGS"].clear()
    del SV
    gc.collect()
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    json.dump(res, open(p, "w"), indent=1, default=str)
    return res


# ------------------------------------------------------------------ verificacao previa
def conferir():
    SV = ambiente(pasta("r9"))
    print("IMS, referencias do artigo, roda0 (so paradas depois da referencia) e roda original:")
    for seq, ref in (("1st_test", (216, 108)), ("2nd_test", (216, 108)), ("4th_test", (144, 6))):
        a = avalia_ims(SV, seq, ref)
        b = avalia_ims(SV, seq, ref, roda=lambda s, bb, s0, n: SV["roda"](s, bb, s0, n, K8))
        print("  %s roda0: rol %s onset %s lead %s | original: rol %s onset %s lead %s | alarmes roda0 %s | original %s"
              % (seq, a["primeiro"], a["onset"], a["lead_d"], b["primeiro"], b["onset"], b["lead_d"],
                 a["alarmes"], b["alarmes"]), flush=True)
    print("PRONOSTIA, (30, 30):")
    on = {bn: SV["pron_res"](bn, 30, 30)["onset"] for bn in SV["PRON"]}
    print("  ", on, flush=True)
    print("JNU, 0 + 3 s:")
    for rpm in (600, 800, 1000):
        d = jnu_desfecho(SV, rpm, 0, int(round(SV["PISO_SINAL"] * SV["WPS_J"])))
        print("  %d rpm: %d de 4 corretas | %s" % (rpm, d["corretas"], {k: (v["k"], v["kmin"]) for k, v in d.items()
                                                                       if k != "corretas"}), flush=True)


# ------------------------------------------------------------------ teste de corrida (Sec. 7)
D_HORAS = (4, 24, 48, 72, 96)                          # 4 h declarado (API 610); os outros, sensibilidade


def teste_corrida(SV, seq, D):
    """Regime nominal do IMS: referencia = os D mais recentes; trava no primeiro W(t) estavel;
    recusa se nao estabilizar ate 2D; paradas nao zeram o comissionamento."""
    dados, onset = SV["dados"], SV["onset"]
    U = {b: dados(seq, b) for b in (1, 2, 3, 4)}
    day = U[1][2]
    total, bloco = len(day), 3
    s0 = 0
    t = s0 + D
    testes = 0
    while True:
        if t > total:
            return dict(trava=None, motivo="fim dos dados", testes=testes)
        if t - s0 > 2 * D:
            return dict(trava=None, motivo="nao estabilizou ate 2D (recusa)", testes=testes, t_recusa=int(t))
        testes += 1
        if not any(alarme_na_segunda_metade(U[b][0], U[b][1], U[b][4], t - D, D, onset) for b in U):
            return dict(trava=(int(t - D), int(D)), motivo="estavel", testes=testes,
                        fim_comissionamento_dia=float(day[t - 1]), fim_comissionamento_h_aquisicao=t / 6.0)
        t += bloco


def roda_corrida(tag):
    p = os.path.join(OUT, tag, "resultado.json")
    if not os.path.exists(p):
        return None
    res = json.load(open(p))
    SV = ambiente(pasta(tag))
    out = {}
    for Dh in D_HORAS:
        out[str(Dh)] = {}
        for seq in SV["SEQS"]:
            m = teste_corrida(SV, seq, Dh * 6)
            m["desfecho"] = avalia_ims(SV, seq, m["trava"])
            if m["trava"] and m["desfecho"].get("onset") is not None:
                m["dias_entre_trava_e_alarme"] = round(float(SV["dados"](seq, 1)[2][m["desfecho"]["onset"]])
                                                       - m["fim_comissionamento_dia"], 2)
            out[str(Dh)][seq] = m
        print("%s D=%d h pronto" % (tag, Dh), flush=True)
    res["corrida"] = out
    json.dump(res, open(p, "w"), indent=1, default=str)
    SV["CACHE"].clear()
    SV["FLAGS"].clear()
    del SV
    gc.collect()
    return res


def resumo_corrida():
    linhas = []
    for tag in ("r9",) + REPRES:
        p = os.path.join(OUT, tag, "resultado.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if "corrida" not in r:
            continue
        for Dh in D_HORAS:
            R = r["corrida"][str(Dh)]
            cel = []
            certos = 0
            for s in ("1st_test", "2nd_test", "4th_test"):
                m = R[s]
                if not m["trava"]:
                    cel.append("%s (%s)" % ("recusa" if "recusa" in m["motivo"] else "sem trava", m["motivo"][:20]))
                    continue
                d = m["desfecho"]
                fim = "trava dia %.2f" % m["fim_comissionamento_dia"]
                if d.get("primeiro") is None:
                    cel.append("%s, sem alarme" % fim)
                    continue
                certos += int(d["certo"])
                cel.append("%s; rol %d%s %.2f d" % (fim, d["primeiro"], "" if d["certo"] else "*", d["lead_d"]))
            linhas.append("%-6s D=%2d h  IMS %d/3 | %s" % (tag, Dh, certos, " | ".join(cel)))
        linhas.append("")
    txt = "\n".join(linhas)
    io.open(os.path.join(OUT, "resumo_corrida.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


# ------------------------------------------------------------------ dois regimes (Sec. 6)
MAF_N1 = os.path.join(H, "final_r10r11", "resumo.json")


def regimes(tag):
    """Regime acelerado no PRONOSTIA: referencia fixa (0, 30), o minimo de 3 s no inicio. O regime
    nominal (IMS, JNU) e a variante "var" ja gravada; o MAFAULDA vem de final_r10r11."""
    p = os.path.join(OUT, tag, "resultado.json")
    if not os.path.exists(p):
        return None
    res = json.load(open(p))
    SV = ambiente(pasta(tag))
    pr = {}
    for bn in SV["PRON"]:
        x = SV["pron_res"](bn, 0, 30)
        pr[bn] = None if x is None else dict(
            onset=x["onset"], rem=None if x["rem"] is None else round(x["rem"], 2),
            lead_h=None if x["lead_h"] is None else round(x["lead_h"], 3), n=x["n"], em_10min=x["val"], cond=x["cond"])
    res["regimes"] = dict(pron_acelerado=pr, ref=(0, 30))
    json.dump(res, open(p, "w"), indent=1, default=str)
    del SV
    gc.collect()
    return res


def resumo_regimes():
    maf = json.load(open(MAF_N1, encoding="utf-8")) if os.path.exists(MAF_N1) else {}
    linhas = ["| Repr. | MAFAULDA | IMS Set 1 | Set 2 | Set 3 | IMS | PRONOSTIA | mediana % c1/c2/c3 | em 10 min | JNU |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in ("r9",) + REPRES:
        p = os.path.join(OUT, tag, "resultado.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if "regimes" not in r:
            continue
        R = r["var"]
        ims = [fmt_ims(R["ims"][s]).split(", ", 1)[-1] for s in ("1st_test", "2nd_test", "4th_test")]
        certos = sum(1 for s in R["ims"].values() if s["trava"] and s["desfecho"].get("certo"))
        pr = r["regimes"]["pron_acelerado"]
        ok = {bn: m for bn, m in pr.items() if m and m["onset"] is not None}
        med = "/".join("%.0f" % np.median([m["rem"] for bn, m in ok.items() if m["cond"] == c]) if any(
            m["cond"] == c for m in ok.values()) else "-" for c in (1, 2, 3))
        v10 = sum(1 for m in ok.values() if m["em_10min"])
        jc = sum(m["desfecho"]["corretas"] for m in R["jnu"].values() if m["trava"])
        mf = (maf.get("r9" if tag == "zer" else tag) or {}).get("mafaulda") or {}
        mt = "%d (fp %d)" % (mf["tp"], mf["fp"]) if "tp" in mf else None
        linhas.append("| %s | %s | %s | %s | %s | %d/3 | %d/17 | %s | %d | %d/12 |"
                      % (tag, mt if mt is not None else "?", *ims, certos, len(ok), med, v10, jc))
    txt = "\n".join(linhas)
    io.open(os.path.join(OUT, "resumo_regimes.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


# ------------------------------------------------------------------ resumo
def fmt_ims(m):
    if not m["trava"]:
        return "sem trava"
    s0, n = m["trava"]
    d = m["desfecho"]
    ref = "%g+%g h" % (s0 * 10 / 60.0, n * 10 / 60.0)
    if d.get("primeiro") is None:
        return "%s, sem alarme" % ref
    return "%s, rol %d%s %.2f d" % (ref, d["primeiro"], "" if d["certo"] else "*", d["lead_d"])


def resumo():
    linhas = []
    for tag in ("r9",) + REPRES + tuple("d%d" % i for i in range(5)):
        p = os.path.join(OUT, tag, "resultado.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        for var in VARIANTES:
            if var not in r:
                continue
            R = r[var]
            ims = " | ".join("%s: %s" % (s.replace("_test", ""), fmt_ims(R["ims"][s])) for s in ("1st_test", "2nd_test", "4th_test"))
            certos = sum(1 for s in R["ims"].values() if s["trava"] and s["desfecho"].get("certo"))
            pr = R["pron"]
            alarm = [bn for bn, m in pr.items() if m["trava"] and m.get("desfecho") and m["desfecho"]["onset"] is not None]
            semtrava = [bn for bn, m in pr.items() if not m["trava"]]
            refs = sorted({tuple(m["trava"]) for m in pr.values() if m["trava"]})
            rem = [m["desfecho"]["rem"] for m in pr.values() if m["trava"] and m.get("desfecho") and m["desfecho"]["rem"] is not None]
            jn = R["jnu"]
            jc = sum(m["desfecho"]["corretas"] for m in jn.values() if m["trava"])
            jref = {rpm: (m["trava"][1] if m["trava"] else None) for rpm, m in jn.items()}
            linhas.append("%-6s %-5s IMS %d/3 | %s" % (tag, var, certos, ims))
            linhas.append("%-12s PRONOSTIA %d/17 alarmam, %d sem trava %s, %% restante mediano %.0f (min %.0f); refs (s0,n) %s"
                          % ("", len(alarm), len(semtrava), semtrava, np.median(rem) if rem else np.nan,
                             min(rem) if rem else np.nan, refs[:8]))
            linhas.append("%-12s JNU %d/12 corretas; n travado (janelas) %s" % ("", jc, jref))
        linhas.append("")
    txt = "\n".join(linhas)
    io.open(os.path.join(OUT, "resumo.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    if "conferir" in args:
        conferir()
    so = [a[3:].split(",") for a in args if a.startswith("so=")]
    if "regimes" in args:
        for tag in ("r9",) + REPRES:
            regimes(tag)
            print("%s regimes pronto" % tag, flush=True)
        resumo_regimes()
    if "corrida" in args:
        for tag in ("r9",) + REPRES:
            roda_corrida(tag)
        resumo_corrida()
    for tag in [a for a in args if a not in ("conferir", "resumo", "regimes", "corrida") and not a.startswith("so=")]:
        roda_tag(tag, tuple(so[0]) if so else VARIANTES)
    if "resumo" in args:
        resumo()
