# -*- coding: utf-8 -*-
"""
Secao 9 de resultados-nivel0.md: referencia de 4 h ou 4 h + 4 h no IMS, com os niveis L7 (artigo),
L11 (fixo, so IMS) e Loos (derivado da referencia: taxa fora da amostra entre as metades da
referencia, k_min binomial com o mesmo alfa/(BM), teto em N_r). O resto das bases com Loos.

    cd code && python nivel_oos.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
sys.argv = sys.argv[:1]
import comissionamento_n0 as c  # noqa: E402

ALPHA, Q = 0.05, 0.01
TAGS = ("r9",) + c.REPRES
REFS = {"A 0+4h": (0, 24), "B 4h+4h": (24, 24)}
B_ENROL = (98, 391)
OUT = os.path.join(H, "final_n0")


def kmin_q(nr, q, B, M):
    a = ALPHA / (B * M)
    return next((k for k in range(1, nr + 1) if binom.sf(k - 1, nr, q) <= a), nr)


def q_oos(E, a, h, b):
    mu, var, T = c.modelo(E, a, h)
    return max(Q, float((c.pontua(E[h:b], mu, var) > T).mean()))


# ------------------------------------------------------------------ IMS
def ims(SV):
    res = {}
    for nome, ref in REFS.items():
        s0, n = ref
        for seq in SV["SEQS"]:
            base = {b: SV["dados"](seq, b) for b in (1, 2, 3, 4)}
            qs = {b: q_oos(t[0], t[1][s0], t[1][s0 + n // 2], t[1][s0 + n]) for b, t in base.items()}
            kos = {b: kmin_q(39, qs[b], 4, SV["M_IMS"]) for b in base}
            for nivel, km in (("L7", {b: base[b][4] for b in base}), ("L11", {b: 11 for b in base}), ("Loos", kos)):
                for b, t in base.items():
                    SV["CACHE"][(seq, b)] = (t[0], t[1], t[2], t[3], km[b], t[5])
                SV["FLAGS"].clear()
                d = c.avalia_ims(SV, seq, ref)
                res.setdefault(nome, {}).setdefault(nivel, {})[seq] = dict(
                    desfecho=d, kmin={int(b): int(v) for b, v in km.items()},
                    q_oos={int(b): round(v, 4) for b, v in qs.items()} if nivel == "Loos" else None)
            for b, t in base.items():
                SV["CACHE"][(seq, b)] = t
            SV["FLAGS"].clear()
    return res


# ------------------------------------------------------------------ PRONOSTIA (0, 30)
def pron(SV):
    out = {}
    for bn in SV["PRON"]:
        E, off, km0 = c.pron_dados(SV, bn)
        ns = len(off) - 1
        q = q_oos(E, off[0], off[15], off[30])
        km = kmin_q(4, q, 1, SV["M_PRON"])
        mu, var, T = c.modelo(E, off[0], off[30])
        k = np.add.reduceat((c.pontua(E, mu, var) > T).astype(np.int64), off[:-1])
        i = SV["onset"](k >= km, 30, c.K8)
        out[bn] = dict(q_oos=round(q, 4), kmin=int(km), kmin_artigo=int(km0), onset=None if i is None else int(i),
                       rem=None if i is None else round(100.0 * (ns - i) / ns, 2))
    return out


# ------------------------------------------------------------------ JNU (referencia da regra de crescimento)
def jnu(SV, trava):
    EJ, OJ, LAB, RPM = SV["EJ"], SV["OJ"], SV["LABJ"], SV["RPMJ"]
    out = {}
    for rpm, (a, n) in trava.items():
        i_n = [i for i in range(12) if LAB[i] == "normal" and RPM[i] == int(rpm)][0]
        En = EJ[OJ[i_n]:OJ[i_n + 1]]
        meio = len(En) // 2
        q = q_oos(En, a, a + n // 2, a + n)
        mu, var, T = c.modelo(En, a, a + n)
        r = dict(q_oos=round(q, 4))
        cert = 0
        for nome, X in [("normal", En[meio:])] + [(o, EJ[OJ[i]:OJ[i + 1]]) for o in ("inner", "outer", "ball")
                                                 for i in range(12) if LAB[i] == o and RPM[i] == int(rpm)]:
            kk = int((c.pontua(X, mu, var) > T).sum())
            km = kmin_q(len(X), q, 1, SV["M_JNU"])
            ok = (kk >= km) == (nome != "normal")
            cert += ok
            r[nome] = (kk, len(X), km, bool(ok))
        r["corretas"] = cert
        out[rpm] = r
    return out


# ------------------------------------------------------------------ MAFAULDA (bloco de enrolment)
def maf(SV, tag):
    t = "r9" if tag == "zer" else tag
    csv = os.path.join(H, "final_r10r11", t, "mafaulda_%s.csv" % t)
    emb = os.path.join(H, "rodada9", "mafaulda_emb_r9.npz") if t == "r9" else \
        os.path.join(c.DADOS, "rodada12" if t in c.R12 else "rodada11", "mafaulda_emb_%s.npz" % t)
    if not (os.path.exists(csv) and os.path.exists(emb)):
        return None
    R = pd.read_csv(csv)
    z = np.load(emb, allow_pickle=True)
    E, off, rel = z["E"], z["offsets"], list(z["relpath"])
    qs = {}
    for _, r in R[R.grupo == "normal"].iterrows():
        i = rel.index(r.relpath)
        X = np.asarray(E[off[i]:off[i + 1]], np.float32)
        a, b = B_ENROL
        qs[int(r.ponto)] = q_oos(X, a, a + (b - a) // 2, b)
    M = SV["M_JNU"]
    chk = all(kmin_q(int(n), Q, 1, M) == int(k) for n, k in zip(R.n_win, R.kmin))
    R["kmin_oos"] = [kmin_q(int(n), qs[int(p)], 1, M) for n, p in zip(R.n_win, R.ponto_modelo)]
    R["falha_oos"] = R.flagged >= R.kmin_oos
    nor, op = R[R.grupo == "normal"], R[R.grupo == "open-set"]
    return dict(confere_kmin_artigo=bool(chk), q_oos_mediana=round(float(np.median(list(qs.values()))), 4),
                q_oos_faixa=[round(min(qs.values()), 4), round(max(qs.values()), 4)],
                tp_artigo=int(op.falha_kmin.sum()), fp_artigo=int(nor.falha_kmin.sum()),
                tp_oos=int(op.falha_oos.sum()), fp_oos=int(nor.falha_oos.sum()), n_open=int(len(op)),
                kmin_oos_faixa_falha=[int(R[R.grupo == "open-set"].kmin_oos.min()), int(R[R.grupo == "open-set"].kmin_oos.max())])


def fmt(d):
    if d.get("primeiro") is None:
        return "sem alarme"
    al = sorted(d["alarmes"].items(), key=lambda kv: -kv[1][1])
    return " ".join("%s%s%.2f" % (b, "F" if d["primeiro"] == int(b) and d["certo"] else ("*" if False else ""), v[1])
                    for b, v in al[:2])


if __name__ == "__main__":
    linhas = []
    for tag in TAGS:
        p = os.path.join(OUT, tag, "resultado.json")
        r = json.load(open(p))
        SV = c.ambiente(c.pasta(tag))
        res = dict(ims=ims(SV), pron=pron(SV),
                   jnu=jnu(SV, {rpm: m["trava"] for rpm, m in r["var"]["jnu"].items()}), maf=maf(SV, tag))
        r["nivel_oos"] = res
        json.dump(r, open(p, "w"), indent=1, default=str)
        for nome in REFS:
            for nivel in ("L7", "L11", "Loos"):
                R = res["ims"][nome][nivel]
                cel, cert = [], 0
                for s in ("1st_test", "2nd_test", "4th_test"):
                    d = R[s]["desfecho"]
                    if d.get("primeiro") is None:
                        cel.append("sem alarme")
                        continue
                    cert += int(d["certo"])
                    cel.append("%d%s %.2f" % (d["primeiro"], "" if d["certo"] else "*", d["lead_d"]))
                km = "" if nivel != "Loos" else " kmin %s" % {s[:3]: sorted(set(R[s]["kmin"].values())) for s in R}
                linhas.append("%-6s %-8s %-4s IMS %d/3 | %s%s" % (tag, nome, nivel, cert, " | ".join(cel), km))
        pr = res["pron"]
        al = [m for m in pr.values() if m["onset"] is not None]
        linhas.append("%-6s PRONOSTIA Loos: %d/17, kmin %s, q_oos %.3f-%.3f, mediana rem %.0f"
                      % (tag, len(al), sorted({m["kmin"] for m in pr.values()}),
                         min(m["q_oos"] for m in pr.values()), max(m["q_oos"] for m in pr.values()),
                         np.median([m["rem"] for m in al]) if al else np.nan))
        jn = res["jnu"]
        linhas.append("%-6s JNU Loos: %d/12 | %s" % (tag, sum(v["corretas"] for v in jn.values()),
                                                     "; ".join("%s q=%.3f norm %d/%d (k %d) inner %d/%d (k %d)"
                                                               % (rpm, v["q_oos"], v["normal"][0], v["normal"][1], v["normal"][2],
                                                                  v["inner"][0], v["inner"][1], v["inner"][2])
                                                               for rpm, v in jn.items())))
        mf = res["maf"]
        linhas.append("%-6s MAFAULDA Loos: %s" % (tag, mf))
        linhas.append("")
        print("\n".join(linhas[-9:]), flush=True)
        SV["CACHE"].clear()
        SV["FLAGS"].clear()
        del SV
    io.open(os.path.join(OUT, "resumo_nivel_oos.txt"), "w", encoding="utf-8").write("\n".join(linhas))
