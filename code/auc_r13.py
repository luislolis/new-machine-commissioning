# -*- coding: utf-8 -*-
"""
Rodada 13, etapa 1 (plano-rodada13.md, Secao 4): AUC e pAUC (p = 0,1) da NOSSA cadeia, nos embeddings
que ja estao no disco, com os rotulos declarados.

Escore de uma gravacao ou aquisicao = media dos escores de Mahalanobis (vetor de variancias) das
janelas, sob o modelo da unidade ajustado na referencia da regra B:
  MAFAULDA   bloco de enrolment da normal de cada velocidade (ponto de operacao)
  IMS        aquisicoes (0, 24) de cada rolamento (4 h)
  PRONOSTIA  aquisicoes 0-29 (3 s)
  JNU        primeira metade da gravacao normal de cada rotacao (1465 janelas)
AUC = sklearn.roc_auc_score; pAUC = roc_auc_score(..., max_fpr=0.1) (padronizada, como no avaliador do
DCASE). Anomalo = positivo. Media harmonica entre unidades.

Rotulos (Secao 4):
  MAFAULDA   normais: os 49 blocos de teste (um por velocidade); anomalas: as 1538 gravacoes de categorias
             abertas, cada uma com o modelo da sua velocidade (a normal mais proxima se o ponto nao tem normal).
             Por velocidade, media harmonica e conjunto inteiro ("pooled"). Alem do escore medio, a AUC de
             P_r (fracao de janelas acima de T), que e a do artigo (0,9895 no r9).
  JNU        por rotacao: normal = a metade separada; anomalas = as 3 gravacoes de falha.
  IMS        SUBSTITUTA: normais = as 144 aquisicoes (24 h) depois da referencia; anomalas = os ultimos 10 % da
             corrida dos rolamentos documentados como falhos.
  PRONOSTIA  SUBSTITUTA: normais = as 60 aquisicoes depois da referencia; anomalas = os ultimos 10 % da corrida
             dos 17 rolamentos.

Escolhas que o plano nao fixa (declaradas aqui):
  - "24 h" no IMS = 144 aquisicoes (6 por hora, convencao do codigo); ultimos 10 % = round(0,1 N) aquisicoes.
  - PRONOSTIA truncado: os 10 % valem para as aquisicoes disponiveis (como final_v10 usa ns).
  - Alem do declarado, no JNU vai a AUC por janela (complementar), porque por gravacao ha 1 normal contra 3.

    cd code && python auc_r13.py [etiquetas...] [--refaz]
"""
import gc
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
import apoio_v10 as ap  # noqa: E402  (antes de comissionamento_n0)
import comissionamento_n0 as c  # noqa: E402

OUT = os.path.join(H, "final_r13")
os.makedirs(OUT, exist_ok=True)
TAGS = ["r9", "d0", "d1", "d2", "d3", "d4", "rnd0", "rnd1", "zer", "ce", "specn", "spec", "feat", "featn",
        "r12", "r9nc", "rnd0nc", "r12s", "r12l", "r12g", "r12p"]
FMAX = 0.1
N_NORM_IMS, FRAC_ANOM = 144, 0.10
N_NORM_PRON = 60
NOME_SET = {"1st_test": "Set 1", "2nd_test": "Set 2", "4th_test": "Set 3"}


# ------------------------------------------------------------------ metricas
def auc_pauc(neg, pos):
    """AUC e pAUC(0,1) com anomalo = positivo. Devolve (auc, pauc) ou (None, None)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return None, None
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    return float(roc_auc_score(y, s)), float(roc_auc_score(y, s, max_fpr=FMAX))


def hmean(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return 0.0 if min(v) <= 0 else float(len(v) / sum(1.0 / x for x in v))


def resume(lista):
    """lista de (auc, pauc) -> dict com media harmonica."""
    return dict(hmean_auc=hmean([a for a, _ in lista]), hmean_pauc=hmean([p for _, p in lista]), n_unidades=len(lista))


def pontua64(X, mu, var):
    D = np.asarray(X, np.float64) - mu
    return np.sqrt((D * D / var).sum(axis=1))


def ajusta(Ref):
    Ref = np.asarray(Ref, np.float64)
    mu, var = Ref.mean(0), Ref.var(0, ddof=1) + 1e-12
    T = float(np.quantile(pontua64(Ref, mu, var), 0.99))
    return mu, var, T


# ------------------------------------------------------------------ MAFAULDA
def bloco_maf(tag):
    src, EN, TE = ap._emb_mafaulda(tag)
    if src is None:
        return None
    z = ap._load(src, allow_pickle=True)
    E, o = z["E"], z["offsets"]
    G = pd.read_csv(os.path.join(H, "mafaulda_rotacoes", "gravacoes.csv"))
    meta = pd.DataFrame(dict(relpath=z["relpath"], classe=z["fault_type"], ponto=z["ponto"].astype(int)))
    meta = meta.merge(G[["relpath", "f", "f_nom", "marcas"]], on="relpath", how="left", validate="one_to_one")
    MOD, linhas = {}, []
    for i, r in meta[meta.classe == "normal"].iterrows():
        X = E[o[i]:o[i + 1]].astype(np.float64)
        mu, var, T = ajusta(X[EN])
        s = pontua64(X[TE], mu, var)
        MOD[int(r.ponto)] = (mu, var, T, float(r.f))
        linhas.append(dict(ponto=int(r.ponto), grupo="normal", media=float(s.mean()), pr=float((s > T).mean())))
    pts = np.array(sorted(MOD))
    fn = np.array([MOD[p][3] for p in pts])
    for i, r in meta[meta.classe != "normal"].iterrows():
        g = ("desbal" if r.classe == "imbalance" else
             ("open-set V" if isinstance(r.marcas, str) and "V" in r.marcas else "open-set"))
        if g != "open-set":
            continue
        p = r.ponto if r.ponto in MOD else int(pts[np.argmin(np.abs(fn - r.f_nom))])
        mu, var, T, _ = MOD[p]
        s = pontua64(E[o[i]:o[i + 1]], mu, var)
        linhas.append(dict(ponto=int(p), grupo="open-set", media=float(s.mean()), pr=float((s > T).mean())))
    R = pd.DataFrame(linhas)
    nor, op = R[R.grupo == "normal"], R[R.grupo == "open-set"]
    out = dict(n_normal=int(len(nor)), n_open=int(len(op)))
    for chave, col in (("escore_medio", "media"), ("P_r", "pr")):
        por = []
        det = {}
        for p in sorted(MOD):
            n_, a_ = nor[nor.ponto == p][col].values, op[op.ponto == p][col].values
            auc, pa = auc_pauc(n_, a_)
            if auc is not None:
                por.append((auc, pa))
                det[str(p)] = dict(n_open=int(len(a_)), auc=auc, pauc=pa)
        pool = auc_pauc(nor[col].values, op[col].values)
        out[chave] = dict(pooled=dict(auc=pool[0], pauc=pool[1]), por_velocidade=det, **resume(por))
    del E, z
    return out


# ------------------------------------------------------------------ JNU
def bloco_jnu(SV):
    EJ, OJ, LAB, RPM = SV["EJ"], SV["OJ"], SV["LABJ"], SV["RPMJ"]
    por, por_jan, det = [], [], {}
    for rpm in (600, 800, 1000):
        i_n = [i for i in range(12) if LAB[i] == "normal" and RPM[i] == rpm][0]
        En = EJ[OJ[i_n]:OJ[i_n + 1]]
        meio = len(En) // 2
        mu, var, T = ajusta(En[:meio])
        sn = pontua64(En[meio:], mu, var)
        falhas = {o: pontua64(EJ[OJ[i]:OJ[i + 1]], mu, var) for o in ("inner", "outer", "ball")
                  for i in range(12) if LAB[i] == o and RPM[i] == rpm}
        auc, pa = auc_pauc([sn.mean()], [v.mean() for v in falhas.values()])
        aj, pj = auc_pauc(sn, np.concatenate(list(falhas.values())))
        por.append((auc, pa))
        por_jan.append((aj, pj))
        det[str(rpm)] = dict(ref_janelas=int(meio), n_normal_janelas=int(len(sn)), auc=auc, pauc=pa,
                             janela_auc=aj, janela_pauc=pj)
    return dict(por_rotacao=det, **resume(por), complementar_por_janela=resume(por_jan))


# ------------------------------------------------------------------ IMS
def bloco_ims(SV):
    unidades, det = [], {}
    pn, pa = [], []
    for seq in SV["SEQS"]:
        for b in sorted(SV["FAILED"][seq]):
            E, off, day = SV["dados"](seq, b)[:3]
            N = len(day)
            mu, var, T = ajusta(E[off[0]:off[24]])
            s = pontua64(E, mu, var)
            m = np.add.reduceat(s, off[:-1]) / np.diff(off)
            na = int(round(FRAC_ANOM * N))
            neg, pos = m[24:24 + N_NORM_IMS], m[N - na:]
            auc, pau = auc_pauc(neg, pos)
            unidades.append((auc, pau))
            pn.append(neg)
            pa.append(pos)
            det["%s b%d" % (NOME_SET[seq], b)] = dict(n_normais=int(len(neg)), n_anomalas=int(len(pos)), auc=auc, pauc=pau)
    pool = auc_pauc(np.concatenate(pn), np.concatenate(pa))
    return dict(substituta=True, por_unidade=det, pooled=dict(auc=pool[0], pauc=pool[1]), **resume(unidades))


# ------------------------------------------------------------------ PRONOSTIA
def bloco_pron(SV):
    unidades, det, pn, pa = [], {}, [], []
    for bn in SV["PRON"]:
        E, off, _ = c.pron_dados(SV, bn)
        ns = len(off) - 1
        mu, var, T = ajusta(E[off[0]:off[30]])
        s = pontua64(E, mu, var)
        m = np.add.reduceat(s, off[:-1]) / np.diff(off)
        na = int(round(FRAC_ANOM * ns))
        neg, pos = m[30:30 + N_NORM_PRON], m[ns - na:]
        auc, pau = auc_pauc(neg, pos)
        unidades.append((auc, pau))
        pn.append(neg)
        pa.append(pos)
        det[bn] = dict(n_aquisicoes=int(ns), n_anomalas=int(len(pos)), auc=auc, pauc=pau)
    pool = auc_pauc(np.concatenate(pn), np.concatenate(pa))
    return dict(substituta=True, por_unidade=det, pooled=dict(auc=pool[0], pauc=pool[1]), **resume(unidades))


# ------------------------------------------------------------------ uma etiqueta
def roda_tag(tag):
    t0 = time.time()
    pasta = ap.pasta_v10(tag)
    if not (pasta.startswith(ap.PREFIXO_ZIP) or os.path.isdir(pasta)):
        print("%s: %s nao existe, pulando" % (tag, pasta), flush=True)
        return None
    res = dict(tag=tag, pasta=ap.relativo(pasta))
    SV = c.ambiente(pasta)
    for nome, f in (("ims", bloco_ims), ("pron", bloco_pron), ("jnu", bloco_jnu)):
        try:
            res[nome] = f(SV)
        except (FileNotFoundError, KeyError, OSError) as e:
            res[nome] = None
            print("   %s: %s sem %s (%s)" % (tag, nome, "dados", e), flush=True)
    SV["CACHE"].clear()
    SV["FLAGS"].clear()
    del SV
    gc.collect()
    try:
        res["mafaulda"] = bloco_maf(tag)
    except (FileNotFoundError, KeyError, OSError) as e:
        res["mafaulda"] = None
        print("   %s: sem MAFAULDA (%s)" % (tag, e), flush=True)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "auc_%s.json" % tag), "w", encoding="utf-8"), indent=1, default=float)
    print("%s pronto (%.0f s)" % (tag, time.time() - t0), flush=True)
    return res


# ------------------------------------------------------------------ texto
def f4(x):
    return "  n/d " if x is None else "%.4f" % x


def junta():
    tudo = {}
    for tag in TAGS:
        p = os.path.join(OUT, "auc_%s.json" % tag)
        if os.path.exists(p):
            tudo[tag] = json.load(open(p, encoding="utf-8"))
    json.dump(tudo, open(os.path.join(OUT, "auc_cadeia.json"), "w", encoding="utf-8"), indent=1)
    L = ["AUC e pAUC (p = 0,1) da cadeia, escore = media dos escores de janela (plano-rodada13.md, Secao 4).",
         "IMS e PRONOSTIA: rotulos SUBSTITUTOS (normais logo depois da referencia vs ultimos 10 % da corrida).",
         "MAFAULDA 'P_r' = AUC da fracao de janelas acima de T (a do artigo). hm = media harmonica entre unidades.",
         ""]
    cab = "%-7s | %-33s | %-33s | %-33s | %-33s" % ("tag", "MAFAULDA escore medio", "MAFAULDA P_r (artigo)", "JNU (3 rotacoes)", "IMS | PRONOSTIA (substitutas)")
    L.append(cab)
    L.append("%-7s | %s | %s | %s | %s" % ("", "hm AUC  hm pAUC  pool AUC  pool pAUC", "hm AUC  hm pAUC  pool AUC  pool pAUC",
                                          "hm AUC   hm pAUC", "IMS hm AUC pAUC | PRON hm AUC pAUC"))
    for tag, r in tudo.items():
        m = r.get("mafaulda")
        if m:
            a = m["escore_medio"]
            b = m["P_r"]
            ma = "%s %s %s %s" % (f4(a["hmean_auc"]), f4(a["hmean_pauc"]), f4(a["pooled"]["auc"]), f4(a["pooled"]["pauc"]))
            mb = "%s %s %s %s" % (f4(b["hmean_auc"]), f4(b["hmean_pauc"]), f4(b["pooled"]["auc"]), f4(b["pooled"]["pauc"]))
        else:
            ma = mb = "sem MAFAULDA"
        j = r.get("jnu")
        jt = "%s %s" % (f4(j["hmean_auc"]), f4(j["hmean_pauc"])) if j else "n/d"
        i, p = r.get("ims"), r.get("pron")
        it = "%s %s" % (f4(i["hmean_auc"]), f4(i["hmean_pauc"])) if i else "n/d"
        pt = "%s %s" % (f4(p["hmean_auc"]), f4(p["hmean_pauc"])) if p else "n/d"
        L.append("%-7s | %s | %s | %s | %s | %s" % (tag, ma, mb, jt, it, pt))
    L.append("")
    for tag, r in tudo.items():
        L.append("--- %s" % tag)
        m = r.get("mafaulda")
        if m:
            L.append("  MAFAULDA: %d normais, %d abertas; velocidades com AUC: %d" % (m["n_normal"], m["n_open"], m["escore_medio"]["n_unidades"]))
        j = r.get("jnu")
        if j:
            L.append("  JNU por rotacao (gravacao: 1 normal vs 3 falhas): " + "; ".join(
                "%s rpm AUC %s pAUC %s" % (k, f4(v["auc"]), f4(v["pauc"])) for k, v in j["por_rotacao"].items()))
            L.append("  JNU por janela (complementar): hm AUC %s hm pAUC %s" % (f4(j["complementar_por_janela"]["hmean_auc"]), f4(j["complementar_por_janela"]["hmean_pauc"])))
        i = r.get("ims")
        if i:
            L.append("  IMS (substituta) por rolamento: " + "; ".join("%s AUC %s pAUC %s" % (k, f4(v["auc"]), f4(v["pauc"])) for k, v in i["por_unidade"].items())
                     + "; pooled AUC %s pAUC %s" % (f4(i["pooled"]["auc"]), f4(i["pooled"]["pauc"])))
        p = r.get("pron")
        if p:
            L.append("  PRONOSTIA (substituta): hm AUC %s pAUC %s; pooled AUC %s pAUC %s; menor AUC por rolamento %s"
                     % (f4(p["hmean_auc"]), f4(p["hmean_pauc"]), f4(p["pooled"]["auc"]), f4(p["pooled"]["pauc"]),
                        f4(min(v["auc"] for v in p["por_unidade"].values()))))
    io.open(os.path.join(OUT, "auc_cadeia.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("auc_cadeia.json e auc_cadeia.txt: %d etiquetas" % len(tudo), flush=True)


if __name__ == "__main__":
    refaz = "--refaz" in ARGS
    tags = [a for a in ARGS if not a.startswith("--")] or TAGS
    for tag in tags:
        if os.path.exists(os.path.join(OUT, "auc_%s.json" % tag)) and not refaz:
            print("%s: ja existe, pulando" % tag, flush=True)
            continue
        roda_tag(tag)
        gc.collect()
    junta()
