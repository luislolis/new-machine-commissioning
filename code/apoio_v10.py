# -*- coding: utf-8 -*-
"""
Funcoes de apoio da versao final (plano-fechamento-v10.md), testadas em 2026-11-13 na sessao que
avaliou a rodada 12. Importe ANTES de comissionamento_n0 quando for ler as dobras dos zips:

    import apoio_v10 as ap            # instala o leitor de zips no np.load
    import comissionamento_n0 as c
    SV = c.ambiente(ap.pasta_v10("d0"))

O que tem aqui:
  pasta_v10(tag)            pasta (ou pseudo-pasta de zip) dos embeddings de cada etiqueta
  mafaulda(tag)             MAFAULDA no nivel do artigo (q = 0,01) e no nivel Loos, com os blocos certos
  linha_do_tempo(S, b, n)   custo em inspecoes: recomissiona depois de cada alarme e segue ate o fim
"""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import binom

H = os.path.dirname(os.path.abspath(__file__))
DADOS = os.environ.get("MISAEL_DADOS") or H     # data next to the scripts unless MISAEL_DADOS says otherwise
K = 7
PREFIXO_ZIP = "ZIP10::"

# ------------------------------------------------------------------ leitor de zips (dobras d0..d4)
_load = np.load
_ZIPS = {}


def _np_load(path, *a, **k):
    """np.load que tambem le "ZIP10::<tag>/<caminho>" de code/rerun10_<tag>.zip, sem descompactar
    (o D: com a rodada 10 descompactada nem sempre esta ligado, e o C: nao tem espaco)."""
    p = str(path)
    if p.startswith(PREFIXO_ZIP):
        tag, memb = p[len(PREFIXO_ZIP):].replace("\\", "/").split("/", 1)
        if tag not in _ZIPS:
            _ZIPS[tag] = zipfile.ZipFile(os.path.join(H, "rerun10_%s.zip" % tag))
        return _load(io.BytesIO(_ZIPS[tag].read(tag + "/" + memb)), *a, **k)
    return _load(path, *a, **k)


np.load = _np_load

DOBRAS = ("d0", "d1", "d2", "d3", "d4")
REPRES11 = ("rnd0", "rnd1", "zer", "ce", "specn", "spec", "feat", "featn")
REPRES12 = ("r12", "r9nc", "rnd0nc", "r12s", "r12l", "r12g", "r12p")
REPRES13P = ("r9p16", "rnd0p16")        # rede projetada em 16 por PCA fixa (pca16_r13.py, 2026-11-16)
# tripletos com todas as familias do MAFAULDA: binario e multiclasse, sementes 42-44 (plano-rodada14.md)
REPRES14 = tuple("r14%s%d" % (o, k) for o in ("b", "m") for k in range(3))


def relativo(p):
    """Caminho de dados como fica gravado nos resultados: relativo a code/ quando possivel."""
    if p.startswith(PREFIXO_ZIP) or not os.path.isabs(p):
        return p
    try:
        r = os.path.relpath(p, H)
    except ValueError:                   # outro disco
        return p
    return p if r.startswith("..") else r.replace("\\", "/")


def pasta_v10(tag):
    if tag == "r9":
        return os.path.join(H, "rerun9")
    if tag in DOBRAS:
        d = os.path.join(DADOS, "rerun10", tag)
        return d if os.path.isdir(d) else PREFIXO_ZIP + tag
    if tag in REPRES11:
        return os.path.join(DADOS, "rerun11", tag)
    if tag in REPRES12:
        return os.path.join(DADOS, "rerun12", tag)
    if tag in REPRES13P:
        return os.path.join(DADOS, "rerun13pca", tag)
    if tag in REPRES14:
        return os.path.join(DADOS, "rerun14", tag)
    raise ValueError(tag)


# ------------------------------------------------------------------ MAFAULDA
B_ENROL, B_TESTE = np.arange(98, 391), np.arange(392, 487)
M_ANO = 365 * 24 * 3600 / 10.0           # o horizonte do artigo para o MAFAULDA (k_min(95) = 11)


def kmin_q(n, q, M=M_ANO):
    a = 0.05 / M
    return next((k for k in range(1, n + 1) if binom.sf(k - 1, n, q) <= a), n)


def _score(X, mu, var):
    D = np.asarray(X, np.float64) - mu
    return np.sqrt((D * D / var).sum(axis=1))


def _emb_mafaulda(tag):
    """(caminho ou bytes do npz, blocos de enrolment e teste)."""
    if tag in ("r9", "zer"):
        return os.path.join(H, "rodada9", "mafaulda_emb_r9.npz"), B_ENROL, B_TESTE
    if tag in DOBRAS:
        z = zipfile.ZipFile(os.path.join(H, "rodada10_treino.zip"))
        b = json.loads(z.read("particao.json"))["dobras"][tag[1:]]
        return io.BytesIO(z.read("mafaulda_emb_%s.npz" % tag)), np.array(b["enrolment"]), np.array(b["teste"])
    if tag in REPRES11:
        return os.path.join(DADOS, "rodada11", "mafaulda_emb_%s.npz" % tag), B_ENROL, B_TESTE
    if tag in REPRES12 and tag != "r12p":
        return os.path.join(DADOS, "rodada12", "mafaulda_emb_%s.npz" % tag), B_ENROL, B_TESTE
    if tag in REPRES13P:
        return os.path.join(DADOS, "rodada13pca", "mafaulda_emb_%s.npz" % tag), B_ENROL, B_TESTE
    if tag in REPRES14:      # as falhas foram vistas no treino: so a especificidade das normais e teste
        return os.path.join(DADOS, "rodada14", "mafaulda_emb_%s.npz" % tag), B_ENROL, B_TESTE
    return None, None, None


def mafaulda(tag):
    """Como avalia_novo (analise_final_v6.py): modelo de cada ponto no enrolment da sua normal;
    gravacoes de ponto sem normal usam a normal de rotacao nominal mais proxima. Nivel do artigo
    (q = 0,01) e Loos (q medido entre as metades do enrolment, piso 0,01). Reproduz o r9:
    1505/1538 e 1500 (Loos), 0/49 normais."""
    src, EN, TE = _emb_mafaulda(tag)
    if src is None:
        return None
    z = _load(src, allow_pickle=True)
    E, o = z["E"], z["offsets"]
    G = pd.read_csv(os.path.join(H, "mafaulda_rotacoes", "gravacoes.csv"))
    meta = pd.DataFrame(dict(relpath=z["relpath"], classe=z["fault_type"], ponto=z["ponto"].astype(int)))
    meta = meta.merge(G[["relpath", "f", "f_nom", "marcas"]], on="relpath", how="left", validate="one_to_one")
    MOD, rows = {}, []
    for i, r in meta[meta.classe == "normal"].iterrows():
        X = E[o[i]:o[i + 1]].astype(np.float64)
        mu, var = X[EN].mean(0), X[EN].var(0, ddof=1) + 1e-12
        T = float(np.quantile(_score(X[EN], mu, var), 0.99))
        h = len(EN) // 2
        mu2, var2 = X[EN[:h]].mean(0), X[EN[:h]].var(0, ddof=1) + 1e-12
        T2 = float(np.quantile(_score(X[EN[:h]], mu2, var2), 0.99))
        q = max(0.01, float((_score(X[EN[h:]], mu2, var2) > T2).mean()))
        MOD[r.ponto] = (mu, var, T, r.f, q)
        s = _score(X[TE], mu, var)
        rows.append(dict(relpath=r.relpath, classe="normal", grupo="normal", n=len(s), k=int((s > T).sum()), q=q))
    pts = np.array(sorted(MOD))
    fn = np.array([MOD[p][3] for p in pts])
    for i, r in meta[meta.classe != "normal"].iterrows():
        p = r.ponto if r.ponto in MOD else int(pts[np.argmin(np.abs(fn - r.f_nom))])
        mu, var, T, _, q = MOD[p]
        s = _score(E[o[i]:o[i + 1]], mu, var)
        g = ("desbal" if r.classe == "imbalance" else
             ("open-set V" if isinstance(r.marcas, str) and "V" in r.marcas else "open-set"))
        rows.append(dict(relpath=r.relpath, classe=r.classe, grupo=g, n=len(s), k=int((s > T).sum()), q=q))
    R = pd.DataFrame(rows)
    R["alarme_artigo"] = [k >= kmin_q(n, 0.01) for k, n in zip(R.k, R.n)]
    R["alarme_loos"] = [k >= kmin_q(n, q) for k, n, q in zip(R.k, R.n, R.q)]
    op, nor = R[R.grupo == "open-set"], R[R.grupo == "normal"]
    res = dict(n_open=int(len(op)), n_normal=int(len(nor)),
               tp_artigo=int(op.alarme_artigo.sum()), fp_artigo=int(nor.alarme_artigo.sum()),
               tp_loos=int(op.alarme_loos.sum()), fp_loos=int(nor.alarme_loos.sum()),
               janelas_normais_pct=float(100 * nor.k.sum() / nor.n.sum()),
               por_classe={c: "%d/%d" % (d.alarme_loos.sum(), len(d)) for c, d in op.groupby("classe")})
    return res, R


# ------------------------------------------------------------------ custo em inspecoes
def linha_do_tempo(S, b, n=24, s0=0):
    """Operacao continua de um rolamento: referencia (s0, n) com o nivel Loos; a cada alarme, a
    maquina e inspecionada e recomissionada (nova referencia de n aquisicoes comecando K depois do
    onset) e o monitoramento segue ate o fim. Nas paradas documentadas, o recomissionamento do
    artigo (aceite por RMS, Delta = 1,5 dB). S = regra_rms.Set(SV, seq).
    Devolve (alarmes em dias antes do fim, numero de recomissionamentos aceitos em paradas).

    O aceite nas paradas e o do artigo (regra_rms.Set.roda): o RMS medio das n aquisicoes depois da
    parada, em dB contra o nivel da referencia de COMISSIONAMENTO (a inicial, ou a do ultimo
    recomissionamento depois de inspecao), tem de ficar dentro de 1,5 dB. Um recomissionamento
    aceito numa parada troca mu, sigma2 e T, mas nao a base do aceite. Assim o primeiro alarme de
    cada rolamento coincide com o de Set.roda (corrigido em 2026-11-16; antes a base seguia a
    referencia em vigor e deixava o aceite acompanhar a deriva, de parada em parada)."""
    SV = S.SV
    E, off, day, db0, _, paradas = SV["dados"](S.seq, b)
    N = len(day)
    fim = float(day[-1])

    def base_db(r0):
        return 20 * np.log10(np.mean(10.0 ** (db0[r0:r0 + n] / 20.0)))

    cur, pos, al, rec = (s0, n), s0 + n, [], 0
    base, km = base_db(s0), S.loos(b, s0, n)
    stops = [a for a in paradas if a >= s0 + n]
    while pos < N:
        prox = [a for a in stops if a > pos]
        fimseg = prox[0] if prox else N
        # numa parada aceita sao reestimados mu, sigma2 e T (Sec. 2.3.5 do artigo); k_min e a base
        # do aceite ficam os do comissionamento, como em Set.roda
        j = SV["onset"](S.contagens(b, *cur)[pos:fimseg] >= km, 0, K)
        if j is not None:
            a = pos + j
            al.append(round(fim - float(day[a]), 2))
            r0 = a + K
            if r0 + n + K > N:
                break
            cur, pos, base, km = (r0, n), r0 + n, base_db(r0), S.loos(b, r0, n)   # comissionamento novo
            continue
        if not prox:
            break
        a = prox[0]
        if a + n + K <= N and abs(float((db0[a:a + n] - base).mean())) <= 1.5:
            cur, pos, rec = (a, n), a + n, rec + 1
        else:
            pos = a
    return al, rec
