# -*- coding: utf-8 -*-
"""Troca, nos .json de resultados, os caminhos absolutos da maquina dos autores pelo caminho relativo a code/
(por exemplo ".../code/rerun9" -> "rerun9", ".../misael_artigo/rerun11/ce" -> "rerun11/ce"). So metadado:
nenhum numero muda.

    python tools/limpa_caminhos.py code/final_v10/*.json code/final_r13/*.json
"""
import io
import json
import re
import sys

PADRAO = re.compile(r"^[A-Za-z]:[\\/].*?[\\/]((?:rerun|rodada)[^\\/]*(?:[\\/].*)?)$")


def limpa(o):
    if isinstance(o, dict):
        return {k: limpa(v) for k, v in o.items()}
    if isinstance(o, list):
        return [limpa(v) for v in o]
    if isinstance(o, str):
        m = PADRAO.match(o)
        if m:
            return m.group(1).replace("\\", "/")
        assert not re.match(r"^[A-Za-z]:[\\/]", o), o
    return o


for p in sys.argv[1:]:
    s = io.open(p, encoding="utf-8").read()
    o = json.loads(s)
    n = limpa(o)
    if n != o:
        ind = 1 if "\n " in s[:200] and "\n  " not in s[:200] else (2 if "\n  " in s[:200] else None)
        io.open(p, "w", encoding="utf-8").write(json.dumps(n, indent=ind))
        print("limpo:", p)
