# -*- coding: utf-8 -*-
"""
Monta os arquivos do registro do Zenodo a partir das pastas de trabalho (uso dos autores; nao faz
parte da reproducao). Os .npz ja sao comprimidos, entao os zips novos usam ZIP_STORED.

    python tools/empacota_zenodo.py <pasta code/ do artigo> <pasta de dados> <pasta de saida>

Zips novos, gravados na pasta de saida (extrair em code/):
    data_r9.zip                       rerun9/, rodada9/ (sem mafaulda_emb_r7.npz), rerun4_artifacts/ (*_feat.npz)
    data_alternative_external.zip     rerun11/{rnd0,rnd1,zer,ce,specn,spec,featn,feat}/
    data_alternative_mafaulda.zip     rodada11/ (embeddings do MAFAULDA, checkpoints, hiper.csv)
Sobem como estao, da pasta code/ (sem copia, por falta de disco): rerun10_d0..d4.zip,
rodada10_treino.zip, rerun14_r14*.zip, rodada14_treino.zip.
Escreve na pasta de saida MANIFEST.txt (SHA-256, bytes, nome) e upload.txt (caminho de cada arquivo a subir).
"""
import hashlib
import os
import sys
import zipfile

CODE, DADOS, SAIDA = sys.argv[1:4]
os.makedirs(SAIDA, exist_ok=True)
REP11 = ("rnd0", "rnd1", "zer", "ce", "specn", "spec", "featn", "feat")
MAF11 = ["mafaulda_emb_%s.npz" % t for t in ("rnd0", "rnd1", "ce", "spec", "specn", "feat", "featn")] + \
        ["embedding_model_r11_%s.pth" % t for t in ("rnd0", "rnd1", "ce")] + ["hiper.csv", "manifest.json", "particao.json"]
COMO_ESTAO = ["rerun10_d%d.zip" % i for i in range(5)] + ["rodada10_treino.zip"] + \
             ["rerun14_r14%s%d.zip" % (o, i) for o in "bm" for i in range(3)] + ["rerun14_r9.zip", "rodada14_treino.zip"]


def arvore(base, prefixo, filtro=lambda rel: True):
    for dp, _, fs in os.walk(base):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, base).replace("\\", "/")
            if filtro(rel):
                yield p, prefixo + "/" + rel


def zipa(nome, itens):
    destino = os.path.join(SAIDA, nome)
    if os.path.exists(destino):
        print("ja existe:", nome)
        return destino
    tmp = destino + ".parcial"
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p, arc in itens:
            if not os.path.exists(p):
                raise SystemExit("falta " + p)
            z.write(p, arc)
            n += 1
    os.replace(tmp, destino)
    print("%-34s %5d arquivos %8.1f MB" % (nome, n, os.path.getsize(destino) / 2**20), flush=True)
    return destino


def itens_r9():
    yield from arvore(os.path.join(CODE, "rerun9"), "rerun9")
    yield from arvore(os.path.join(CODE, "rodada9"), "rodada9", lambda rel: rel != "mafaulda_emb_r7.npz")
    yield from arvore(os.path.join(CODE, "rerun4_artifacts"), "rerun4_artifacts",
                      lambda rel: rel.endswith("_feat.npz") or rel == "manifest.json")


def itens_r11():
    for t in REP11:
        yield from arvore(os.path.join(DADOS, "rerun11", t), "rerun11/" + t)


def itens_maf11():
    for f in MAF11:
        yield os.path.join(DADOS, "rodada11", f), "rodada11/" + f


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


arquivos = [zipa("data_r9.zip", itens_r9()),
            zipa("data_alternative_external.zip", itens_r11()),
            zipa("data_alternative_mafaulda.zip", itens_maf11())]
arquivos += [os.path.join(CODE, f) for f in COMO_ESTAO]
linhas, total = [], 0
for p in arquivos:
    total += os.path.getsize(p)
    linhas.append("%s  %12d  %s" % (sha(p), os.path.getsize(p), os.path.basename(p)))
    print(linhas[-1], flush=True)
open(os.path.join(SAIDA, "MANIFEST.txt"), "w").write("\n".join(linhas) + "\n")
open(os.path.join(SAIDA, "upload.txt"), "w", encoding="utf-8").write("\n".join(arquivos) + "\n")
print("total %.2f GB em %d arquivos" % (total / 2**30, len(arquivos)))
