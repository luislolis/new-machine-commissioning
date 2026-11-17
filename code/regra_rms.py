# -*- coding: utf-8 -*-
"""
Secao 11 de resultados-nivel0.md: estabilizacao pelo RMS (a "temperatura" da API 610) para escolher a
referencia do IMS, e o ponto fixo 12 + 48 h, nas nove representacoes. Nivel Loos, K = 7.

    cd code && python regra_rms.py
"""
import io
import json
import os
import sys

import numpy as np
from scipy.stats import binom

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
sys.argv = sys.argv[:1]
import comissionamento_n0 as c  # noqa: E402

ALPHA, Q, K, D = 0.05, 0.01, 7, 24
OUT = os.path.join(H, "final_n0")


def kmin_q(nr, q, B, M):
    a = ALPHA / (B * M)
    return next((k for k in range(1, nr + 1) if binom.sf(k - 1, nr, q) <= a), nr)


class Set:
    def __init__(self, SV, seq):
        self.SV, self.seq, self.cont = SV, seq, {}

    def contagens(self, b, r0, rn):
        ch = (b, r0, rn)
        if ch not in self.cont:
            E, off = self.SV["dados"](self.seq, b)[:2]
            mu, var, T = c.modelo(E, off[r0], off[r0 + rn])
            self.cont[ch] = np.add.reduceat((c.pontua(E, mu, var) > T).astype(np.int16), off[:-1])
        return self.cont[ch]

    def loos(self, b, s0, n):
        E, off = self.SV["dados"](self.seq, b)[:2]
        h = s0 + n // 2
        mu, var, T = c.modelo(E, off[s0], off[h])
        q = max(Q, float((c.pontua(E[off[h]:off[s0 + n]], mu, var) > T).mean()))
        return kmin_q(int(np.diff(off).max()), q, 4, self.SV["M_IMS"])

    def roda(self, b, s0, n, km):
        SV = self.SV
        E, off, day, db0, _, paradas = SV["dados"](self.seq, b)
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
            j = SV["onset"](self.contagens(b, *cur)[ini:fim] >= km, 0, K)
            if j is not None:
                return float(day[ini + j])
        return None

    def desfecho(self, s0, n):
        SV, seq = self.SV, self.seq
        day = SV["dados"](seq, 1)[2]
        if s0 + n + K > len(day):
            return dict(cat="nao cabe")
        fim = float(day[-1])
        al = {b: self.roda(b, s0, n, self.loos(b, s0, n)) for b in (1, 2, 3, 4)}
        al = {b: v for b, v in al.items() if v is not None}
        if not al:
            return dict(cat="sem alarme")
        p = min(al, key=al.get)
        fal = [al[b] for b in SV["FAILED"][seq] if b in al]
        if p in SV["FAILED"][seq]:
            cat = "certo"
        elif fal and min(fal) - al[p] <= 1.5:
            cat = "eixo"
        else:
            cat = "saudavel longe"
        return dict(cat=cat, rol=int(p), lead_d=round(fim - al[p], 2))

    def estabiliza(self):
        """Primeiro t (aquisicoes, >= 2D) em que as medias de RMS das duas ultimas janelas de 4 h
        diferem por menos de 2 erros padrao, em todos os rolamentos."""
        SV = self.SV
        dbs = [SV["dados"](self.seq, b)[3] for b in (1, 2, 3, 4)]
        total = len(dbs[0])
        for t in range(2 * D, total + 1, 3):
            ok = True
            for db in dbs:
                a, b_ = db[t - 2 * D:t - D], db[t - D:t]
                if abs(b_.mean() - a.mean()) >= 2 * np.sqrt(a.var(ddof=1) / D + b_.var(ddof=1) / D):
                    ok = False
                    break
            if ok:
                return t
        return None


if __name__ == "__main__":
    linhas, res_all = [], {}
    for tag in ("r9",) + c.REPRES:
        SV = c.ambiente(c.pasta(tag))
        res = {}
        for seq in SV["SEQS"]:
            S = Set(SV, seq)
            day = SV["dados"](seq, 1)[2]
            ts = S.estabiliza()
            r = dict(t_s=ts, t_s_h=None if ts is None else ts / 6.0,
                     t_s_dia=None if ts is None else float(day[min(ts, len(day) - 1)]))
            r["fixo_12_48"] = S.desfecho(72, 288)
            if ts is not None:
                r["R1"] = dict(ref=(ts, D), **S.desfecho(ts, D))
                n2 = max(D, ts)
                r["R2"] = dict(ref=(ts, n2), **S.desfecho(ts, n2))
            res[seq] = r
        res_all[tag] = res
        for v in ("fixo_12_48", "R1", "R2"):
            cel, cert, eixo = [], 0, 0
            for seq in SV["SEQS"]:
                d = res[seq].get(v)
                if d is None:
                    cel.append("sem estabilizar")
                    continue
                ref = "" if v == "fixo_12_48" else "%g+%g h: " % (d["ref"][0] / 6.0, d["ref"][1] / 6.0)
                if "lead_d" in d:
                    cel.append("%s%s rol %d %.2f d" % (ref, {"certo": "✓", "eixo": "~", "saudavel longe": "✗"}[d["cat"]],
                                                       d["rol"], d["lead_d"]))
                else:
                    cel.append(ref + d["cat"])
                cert += d["cat"] == "certo"
                eixo += d["cat"] in ("certo", "eixo")
            linhas.append("%-6s %-10s %d/3 (eixo %d/3) | %s" % (tag, v, cert, eixo, " | ".join(cel)))
        linhas.append("%-6s t_s (h de aquisicao): %s" % (tag, {NOME: res[s]["t_s_h"] for NOME, s in
                                                             zip(("Set 1", "Set 2", "Set 3"), SV["SEQS"])}))
        linhas.append("")
        print("\n".join(linhas[-5:]), flush=True)
        SV["CACHE"].clear()
        del SV
    io.open(os.path.join(OUT, "resumo_regra_rms.txt"), "w", encoding="utf-8").write("\n".join(linhas))
    json.dump(res_all, open(os.path.join(OUT, "regra_rms.json"), "w"), indent=1, default=str)
