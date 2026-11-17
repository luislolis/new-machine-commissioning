# -*- coding: utf-8 -*-
"""
Duas representacoes sem nivel, derivadas das da rodada 11 sem voltar ao sinal:

  specn  forma do espectro: a fracao da energia da janela em cada uma das 16 bandas, por canal,
         em log10 (spec - log10 da soma das bandas do canal). Livre de nivel por construcao.
  featn  indicadores de forma: curtose, assimetria e fator de crista por canal (as tres ultimas
         colunas de feat, que vem na ordem log RMS, log pico, curtose, assimetria, crista).

Le e grava na pasta de dados (MISAEL_DADOS, padrao: a pasta deste script):
    rodada11/mafaulda_emb_{spec,feat}.npz  ->  rodada11/mafaulda_emb_{specn,featn}.npz
    rerun11/{spec,feat}/...                ->  rerun11/{specn,featn}/...   (mesmos nomes)
Todas as outras chaves dos npz sao copiadas sem mudanca.

    cd code && python deriva_r12.py
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.environ.get("MISAEL_DADOS") or os.path.dirname(os.path.abspath(__file__))
NB = 16


def specn(E):
    X = E.astype(np.float64)
    N, D = X.shape
    assert D % NB == 0, D
    X = X.reshape(N, D // NB, NB)
    m = X.max(axis=2, keepdims=True)                    # log-soma-exp estavel, base 10
    tot = m + np.log10(np.sum(10.0 ** (X - m), axis=2, keepdims=True))
    return (X - tot).reshape(N, D).astype(np.float32)


def featn(E):
    N, D = E.shape
    assert D % 5 == 0, D
    k = D // 5
    return np.ascontiguousarray(E[:, 2 * k:]).astype(np.float32)


def deriva(src, dst, fn):
    z = np.load(src, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["E"] = fn(z["E"])
    assert len(out["E"]) == len(z["E"]) and np.isfinite(out["E"]).all(), src
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, **out)
    return z["E"].shape, out["E"].shape


n = 0
for base, novo, fn in (("spec", "specn", specn), ("feat", "featn", featn)):
    a, b = deriva(os.path.join(RAIZ, "rodada11", "mafaulda_emb_%s.npz" % base),
                  os.path.join(RAIZ, "rodada11", "mafaulda_emb_%s.npz" % novo), fn)
    print("MAFAULDA %s %s -> %s %s" % (base, a, novo, b), flush=True)
    n += 1
    org = os.path.join(RAIZ, "rerun11", base)
    for dp, _, fs in os.walk(org):
        for f in sorted(fs):
            if not f.endswith("_emb.npz"):
                continue
            rel = os.path.relpath(os.path.join(dp, f), org)
            a, b = deriva(os.path.join(org, rel), os.path.join(RAIZ, "rerun11", novo, rel), fn)
            n += 1
            if "b3" in f or "jnu" in f or "Bearing1_1" in f:
                print("   %-50s %s -> %s" % (rel, a, b), flush=True)
print("%d arquivos derivados em %s" % (n, RAIZ))
