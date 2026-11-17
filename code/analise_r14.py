# -*- coding: utf-8 -*-
"""
Rodada 14 (plano-rodada14.md): tripletos com todas as familias do MAFAULDA. Aplica o criterio H1
declarado antes de rodar e escreve a Secao 5 do plano.

Antes, descompactar:
    rodada14_treino.zip        em code/rodada14/   (mafaulda_emb_r14*.npz)
    rerun14_<tag>.zip          em code/rerun14/    (cada zip traz a pasta <tag>)

    cd code && python analise_r14.py            fidelidade, final_v10 nas 6 etiquetas, AUC, R2, veredito
    cd code && python analise_r14.py --sem-auc  pula a AUC e o R2 (so o veredito)

Um passo pesado de cada vez: final_v10.py e auc_r13.py rodam em sequencia, como subprocessos.
"""
import io
import json
import os

import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
import apoio_v10 as ap  # noqa: E402

TAGS = list(ap.REPRES14)
REF = ["r9", "d0", "d1", "d2", "d3", "d4"]          # as seis rodadas de treino com desbalanceamento
SETS = (("1st_test", "Set 1"), ("2nd_test", "Set 2"), ("4th_test", "Set 3"))
OUT = os.path.join(H, "final_r14")
PLANO = os.path.join(os.path.dirname(H), "plano-rodada14.md")
os.makedirs(OUT, exist_ok=True)
LOG = []


def diz(t=""):
    print(t, flush=True)
    LOG.append(t)


# ------------------------------------------------------------------ dados no disco
def confere_dados():
    faltam = []
    for t in TAGS:
        if not os.path.isdir(ap.pasta_v10(t)):
            faltam.append(ap.pasta_v10(t))
        if not os.path.exists(ap._emb_mafaulda(t)[0]):
            faltam.append(ap._emb_mafaulda(t)[0])
    if faltam:
        raise SystemExit("faltam (descompacte os zips da rodada 14):\n  " + "\n  ".join(faltam))


def fidelidade():
    """O r9 do rerun_extract14 contra code/rerun9: confere que o extrator reproduz o anterior."""
    a, b = os.path.join(H, "rerun9"), os.path.join(ap.DADOS, "rerun14", "r9")
    if not os.path.isdir(b):
        diz("fidelidade: %s nao existe (o r9 nao estava no Drive?)" % b)
        return None
    pares, pior = [], 0.0
    for dp, _, fs in os.walk(b):
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), b)
            if f.endswith("_emb.npz") and os.path.exists(os.path.join(a, rel)):
                pares.append(rel)
    for rel in sorted(pares):
        za, zb = np.load(os.path.join(a, rel)), np.load(os.path.join(b, rel))
        assert np.array_equal(za["offsets"], zb["offsets"]), "offsets diferem em %s" % rel
        dif = float(np.abs(za["E"].astype(np.float32) - zb["E"].astype(np.float32)).max())
        pior = max(pior, dif)
        if dif > 1e-2:
            diz("   %-48s max|dif| %.4f  <<< DIVERGE" % (rel, dif))
    diz("fidelidade rerun14/r9 contra rerun9: %d arquivos, maior diferenca %.2e" % (len(pares), pior))
    return dict(arquivos=len(pares), maior_diferenca=pior)


def roda(script, tags):
    """final_v10.py ou auc_r13.py nas etiquetas que ainda nao tem resultado, um de cada vez."""
    r = subprocess.run([sys.executable, os.path.join(H, script)] + tags, cwd=H)
    if r.returncode:
        raise SystemExit("%s terminou com erro (%d)" % (script, r.returncode))


# ------------------------------------------------------------------ resultados
def resultado(tag):
    return json.load(io.open(os.path.join(H, "final_v10", "resultado_%s.json" % tag), encoding="utf-8"))


def linha(tag):
    R = resultado(tag)
    m = R["mafaulda"]["resumo"]
    ims = {}
    for seq, nome in SETS:
        d = R["ims"][seq]
        de = d.get("desfecho", {})
        ims[nome] = dict(cat=d["cat"], acerto=bool(d["conta_acerto"]), rol=de.get("rol"), lead_d=de.get("lead_d"))
    return dict(tag=tag, maf_tp=m["tp_loos"], maf_n=m["n_open"], maf_fp=m["fp_loos"], maf_normais=m["n_normal"],
                ims=ims, grade=R["grade"]["tres_de_tres"],
                pron=R["pron"]["alarmam"], pron_total=R["pron"]["total"], pron_10min=R["pron"]["nos_10min"],
                jnu=R["jnu"]["corretas"], jnu_total=R["jnu"]["total"], jnu_acc=R["jnu"]["acc_janela_pct"])


def passa(L):
    """H1 para um treino: Set 1 e Set 2 acertam, PRONOSTIA >= 16, JNU 12/12, nenhum bloco normal."""
    return dict(set1=L["ims"]["Set 1"]["acerto"], set2=L["ims"]["Set 2"]["acerto"],
                pron=L["pron"] >= 16, jnu=L["jnu"] == L["jnu_total"], normais=L["maf_fp"] == 0)


SIMB = {"certo": "acerto", "eixo": "eixo", "saudavel longe": "longe", "suspeito": "suspeito"}


def fmt_ims(d):
    lead = "%.2f" % d["lead_d"] if d["lead_d"] is not None else "-"
    return "%s r%s, %s" % (SIMB.get(d["cat"], d["cat"]), d["rol"], lead)


def tabela(linhas):
    L = ["| Treino | MAFAULDA (normais marcadas) | Set 1 | Set 2 | Set 3 | Grade 3/3 | PRONOSTIA (10 min) | JNU (por janela) |",
         "|---|---|---|---|---|---|---|---|"]
    for x in linhas:
        maf = ("%d/%d vistas" if x["tag"].startswith("r14") else "%d/%d") % (x["maf_tp"], x["maf_n"])
        L.append("| %s | %s (%d) | %s | %s | %s | %d | %d/%d (%d) | %d/%d (%.1f %%) |"
                 % (x["tag"], maf, x["maf_fp"], fmt_ims(x["ims"]["Set 1"]), fmt_ims(x["ims"]["Set 2"]),
                    fmt_ims(x["ims"]["Set 3"]), x["grade"], x["pron"], x["pron_total"], x["pron_10min"],
                    x["jnu"], x["jnu_total"], x["jnu_acc"]))
    return L


# ------------------------------------------------------------------ R2 da forma do espectro (descritivo)
def r2_specn(tags):
    """O mesmo metodo de diag_rnd_spec.py (ridge, lambda 1e-3, 50 000 janelas por metade, semente 0),
    com o r9 refeito como conferencia (0,61 no artigo)."""
    RAIZ = ap.DADOS
    ref = np.load(os.path.join(RAIZ, "rodada11", "mafaulda_emb_specn.npz"), allow_pickle=True)
    REL, OFF = ref["relpath"], ref["offsets"]
    rng = np.random.default_rng(0)
    ordem = rng.permutation(len(REL))
    metade = {"ajuste": np.sort(ordem[: len(REL) // 2]), "teste": np.sort(ordem[len(REL) // 2:])}
    IDX = {}
    for nome, recs in metade.items():
        jan = np.concatenate([np.arange(OFF[i], OFF[i + 1]) for i in recs])
        IDX[nome] = np.sort(rng.choice(jan, 50000, replace=False))
    X = {k: np.asarray(ref["E"][v], np.float64) for k, v in IDX.items()}
    del ref

    def amostra(path):
        z = np.load(path, allow_pickle=True)
        rel, off = z["relpath"], z["offsets"]
        if np.array_equal(rel, REL):
            assert np.array_equal(off, OFF)
            mapa = None
        else:
            pos = {r: i for i, r in enumerate(rel)}
            assert set(pos) == set(REL)
            mapa = np.concatenate([np.arange(off[pos[r]], off[pos[r] + 1]) for r in REL])
        E = z["E"]
        return {k: np.asarray(E[v if mapa is None else mapa[v]], np.float64) for k, v in IDX.items()}

    def r2(Xa, Ya, Xt, Yt, lam=1e-3):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        A = np.hstack([(Xa - mu) / sd, np.ones((len(Xa), 1))])
        T = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))])
        G = A.T @ A
        G[:-1, :-1] += lam * len(A) * np.eye(A.shape[1] - 1)
        W = np.linalg.solve(G, A.T @ Ya)
        sse = ((Yt - T @ W) ** 2).sum(0)
        sst = ((Yt - Yt.mean(0)) ** 2).sum(0)
        return float(1 - sse.sum() / sst.sum())

    res = {}
    for t in ["r9"] + tags:
        Y = amostra(ap._emb_mafaulda(t)[0])
        res[t] = r2(X["ajuste"], Y["ajuste"], X["teste"], Y["teste"])
        diz("   R2 da forma do espectro sobre %-6s %.3f" % (t, res[t]))
    return res


# ------------------------------------------------------------------ principal
def main():
    confere_dados()
    diz("Rodada 14: tripletos com todas as familias do MAFAULDA")
    fid = fidelidade()
    faltam = [t for t in TAGS if not os.path.exists(os.path.join(H, "final_v10", "resultado_%s.json" % t))]
    if faltam:
        diz("final_v10.py em %s ..." % ", ".join(faltam))
        roda("final_v10.py", faltam)
    auc, r2 = None, None
    if "--sem-auc" not in ARGS:
        faltam = [t for t in TAGS if not os.path.exists(os.path.join(H, "final_r13", "auc_%s.json" % t))]
        if faltam:
            roda("auc_r13.py", faltam)
        auc = {t: json.load(io.open(os.path.join(H, "final_r13", "auc_%s.json" % t), encoding="utf-8")) for t in TAGS}
        diz("R2 da forma do espectro (specn) sobre o embedding, como na rodada 11:")
        r2 = r2_specn(TAGS)

    ref = [linha(t) for t in REF]
    novos = [linha(t) for t in TAGS]
    veredito = {}
    for obj, nome in (("b", "binario"), ("m", "multiclasse")):
        crit = {L["tag"]: passa(L) for L in novos if L["tag"][3] == obj}
        ok = all(all(c.values()) for c in crit.values())
        set1 = sum(c["set1"] for c in crit.values())
        veredito[obj] = dict(nome=nome, passa=ok, set1_acertos=set1, criterios=crit)

    S = ["## 5. Resultados", "",
         "Gerado por `code/analise_r14.py`. Os critérios são os da Seção 3, declarados antes de rodar.", ""]
    if fid:
        S.append("**Fidelidade do extrator:** r9 do `rerun_extract14` contra `code/rerun9`, %d arquivos, maior "
                 "diferença %.2e." % (fid["arquivos"], fid["maior_diferenca"]))
        S.append("")
    S += ["**Referência:** as seis rodadas de treino com desbalanceamento.", ""] + tabela(ref)
    S += ["", "**Rodada 14:** todas as famílias. No MAFAULDA, as falhas foram vistas no treino; só os blocos "
          "normais marcados valem como teste.", ""] + tabela(novos)
    S += ["", "**Veredito de H1:**"]
    for obj, v in veredito.items():
        falhas = sorted({k for c in v["criterios"].values() for k, ok in c.items() if not ok})
        S.append("- %s: %s. Set 1 acerta em %d de 3%s." % (
            v["nome"], "**passa**" if v["passa"] else "**não passa**", v["set1_acertos"],
            "" if v["passa"] else "; falha em: " + ", ".join(falhas)))
    set3 = [L["tag"] for L in novos if L["ims"]["Set 3"]["acerto"]]
    S.append("- Previsão declarada para o Set 3 (alarme longe nas seis): %s."
             % ("confirmada" if not set3 else "não confirmada; acertam " + ", ".join(set3)))
    if r2:
        S += ["", "**R² da forma do espectro sobre o embedding** (descritivo; no artigo o r9 dá 0,61): "
              + ", ".join("%s %.2f" % kv for kv in r2.items()) + "."]
    if auc:
        S += ["", "**AUCs da cadeia** (descritivo): ver `code/final_r13/auc_cadeia.txt`."]
    S.append("")

    if os.path.exists(PLANO):        # the working plan is not part of the public repository
        txt = io.open(PLANO, encoding="utf-8").read()
        i = txt.index("## 5. Resultados")
        io.open(PLANO, "w", encoding="utf-8").write(txt[:i] + "\n".join(S))
    json.dump(dict(fidelidade=fid, referencia=ref, rodada14=novos, veredito=veredito, r2_specn=r2),
              io.open(os.path.join(OUT, "analise_r14.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    io.open(os.path.join(OUT, "analise_r14.md"), "w", encoding="utf-8").write("\n".join(S))
    diz("\n".join(S))
    diz("escrito em final_r14/analise_r14.md e final_r14/analise_r14.json")


if __name__ == "__main__":
    main()
