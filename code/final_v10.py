# -*- coding: utf-8 -*-
"""
Numeros finais do artigo (plano-fechamento-v10.md, Etapa 1): a regra B em todas as etiquetas.

  IMS         referencia = trava do teste de corrida de 4 h (24 aquisicoes, API 610); nivel Loos, K = 7,
              recomissionamento do artigo nas paradas; grade s0 x n; custo em inspecoes
  PRONOSTIA   regime acelerado, referencia (0, 30), Loos
  JNU         trava da regra de crescimento, Loos
  MAFAULDA    captura unica, bloco de enrolment, Loos (principal) e nivel do artigo

Nenhuma regra nova, nenhuma constante nova: so as funcoes ja existentes.

    cd code && python final_v10.py [etiquetas...] [--refaz]
Sem etiquetas, roda as 20 (pula as que ja tem final_v10/resultado_<tag>.json, salvo --refaz).
Sempre termina juntando os resultado_*.json em final_v10/numeros.json e numeros.txt.
"""
import gc
import io
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
import apoio_v10 as ap  # noqa: E402  (antes de comissionamento_n0: instala o leitor de zips)
import comissionamento_n0 as c  # noqa: E402
import nivel_oos as no  # noqa: E402
import regra_rms as rr  # noqa: E402

OUT = os.path.join(H, "final_v10")
os.makedirs(OUT, exist_ok=True)
TAGS = ["r9", "d0", "d1", "d2", "d3", "d4", "rnd0", "rnd1", "zer", "ce", "specn", "spec", "featn", "feat",
        "r14b0", "r14b1", "r14b2", "r14m0", "r14m1", "r14m2"]          # the 20 rows of Table 8 of the article
S0_H = (0, 2, 4, 6, 12, 24, 36, 48, 72, 96)
N_H = (1, 2, 4, 6, 12, 18, 24, 36, 48, 72)
D_CORRIDA = 24                       # 4 h em aquisicoes (API 610)
SUSPEITO_D = 10.0                    # certo a mais de 10 d do fim nao conta como acerto
NOME = {"1st_test": "Set 1", "2nd_test": "Set 2", "4th_test": "Set 3"}
LOGF = io.open(os.path.join(OUT, "log.txt"), "a", encoding="utf-8")


def diz(t=""):
    print(t, flush=True)
    LOGF.write(t + "\n")
    LOGF.flush()


def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def nan_none(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else x


# ------------------------------------------------------------------ IMS
def bloco_ims(SV):
    out = {}
    for seq in SV["SEQS"]:
        S = rr.Set(SV, seq)
        day = SV["dados"](seq, 1)[2]
        fim, dur, N = float(day[-1]), float(day[-1] - day[0]), len(day)
        m = c.teste_corrida(SV, seq, D_CORRIDA)
        r = dict(seq=seq, motivo=m["motivo"], testes=m["testes"], trava=m["trava"], N=N, fim_dia=fim, dur_dias=dur,
                 falhou=sorted(int(b) for b in SV["FAILED"][seq]))
        if not m["trava"]:
            r["cat"] = "recusa"
            out[seq] = r
            continue
        s0, n = m["trava"]
        r.update(s0=int(s0), n=int(n), fim_com_dia=m["fim_comissionamento_dia"],
                 fim_com_h_aq=m["fim_comissionamento_h_aquisicao"])
        d = S.desfecho(s0, n)
        r["desfecho"] = d
        cat = d["cat"]
        if cat == "certo" and d["lead_d"] > SUSPEITO_D:
            cat = "suspeito"
        r["cat"] = cat
        r["conta_acerto"] = bool(cat == "certo")
        rol = {}
        for b in (1, 2, 3, 4):
            km = int(S.loos(b, s0, n))
            dia = S.roda(b, s0, n, km)
            x = dict(kmin=km, falhou=bool(b in SV["FAILED"][seq]), alarme_dia=dia)
            if dia is not None:
                i = int(np.searchsorted(day, dia))
                lead = fim - float(dia)
                x.update(onset=i, lead_d=round(lead, 2), rem_pct_tempo=round(100.0 * lead / dur, 2),
                         rem_pct_aq=round(100.0 * (N - i) / N, 2))
            else:
                x.update(onset=None, lead_d=None, rem_pct_tempo=None, rem_pct_aq=None)
            tl, rec = ap.linha_do_tempo(S, b, n, s0)
            x.update(linha_do_tempo=tl, recomissionamentos=int(rec))
            rol[str(b)] = x
        r["rol"] = rol
        # grade s0 x n so depende do Set; guardada fora daqui
        out[seq] = r
        r["_S"] = S
    return out


def bloco_grade(SV, ims):
    """Grade s0 x n (Set.desfecho), reaproveitando os Set (e seus caches de contagens) do bloco IMS."""
    Ss = {seq: ims[seq].get("_S") or rr.Set(SV, seq) for seq in SV["SEQS"]}
    cel = {}
    for s0h in S0_H:
        for nh in N_H:
            ds = [Ss[seq].desfecho(s0h * 6, nh * 6) for seq in SV["SEQS"]]
            cel["%d|%d" % (s0h, nh)] = [d["cat"] for d in ds]
    tres = sum(all(x == "certo" for x in v) for v in cel.values())
    eixo = sum(all(x in ("certo", "eixo") for x in v) for v in cel.values())
    por_set = [sum(v[i] == "certo" for v in cel.values()) for i in range(3)]
    return dict(celulas=cel, tres_de_tres=int(tres), tres_pelo_eixo=int(eixo), certas_por_set=por_set, n_celulas=len(cel))


# ------------------------------------------------------------------ PRONOSTIA
def bloco_pron(SV):
    pr = no.pron(SV)
    linhas = {}
    for bn in SV["PRON"]:
        z = np.load(os.path.join(SV["R7"], "A", "pronostia", "pronostia_rep_win_%s_emb.npz" % bn))
        ns, ntr = len(z["offsets"]) - 1, int(z["n_truncated"])
        m = pr[bn]
        i = m["onset"]
        x = dict(cond=int(bn[7]), snapshots=int(ns), truncado_em=ntr if ntr > 0 else None, onset=i, rem_pct=m["rem"],
                 kmin=m["kmin"], q_oos=m["q_oos"])
        if i is None:
            x.update(lead_h=None, lead_corte_h=None, depois_do_corte=False, nos_10min=False)
        else:
            x.update(lead_h=round((ns - i) * 10 / 3600.0, 3), nos_10min=bool(i - 30 < 60))
            if ntr > 0:
                x.update(lead_corte_h=round((ntr - i) * 10 / 3600.0, 3) if i < ntr else None, depois_do_corte=bool(i >= ntr))
            else:
                x.update(lead_corte_h=None, depois_do_corte=False)
        linhas[bn] = x
    al = [v for v in linhas.values() if v["onset"] is not None]
    res = dict(rolamentos=linhas, alarmam=len(al), total=len(linhas),
               sem_alarme=[bn for bn, v in linhas.items() if v["onset"] is None],
               mediana_rem=float(np.median([v["rem_pct"] for v in al])) if al else None,
               mediana_lead_h=float(np.median([v["lead_h"] for v in al])) if al else None,
               nos_10min=int(sum(v["nos_10min"] for v in al)), por_cond={})
    for cnd in (1, 2, 3):
        a = [v for v in al if v["cond"] == cnd]
        res["por_cond"][str(cnd)] = dict(alarmam=len(a), total=sum(v["cond"] == cnd for v in linhas.values()),
                                         mediana_rem=float(np.median([v["rem_pct"] for v in a])) if a else None,
                                         mediana_lead_h=float(np.median([v["lead_h"] for v in a])) if a else None)
    return res


# ------------------------------------------------------------------ JNU
def bloco_jnu(SV):
    mq = {str(rpm): c.jnu_maquina(SV, rpm, "var") for rpm in (600, 800, 1000)}
    trava = {k: v["trava"] for k, v in mq.items()}
    res = dict(trava={k: (None if v is None else [int(v[0]), int(v[1])]) for k, v in trava.items()},
               motivo={k: v["motivo"] for k, v in mq.items()})
    if any(v is None for v in trava.values()):
        res["sem_trava"] = True
        return res
    jn = no.jnu(SV, trava)
    EJ, OJ, LAB, RPM = SV["EJ"], SV["OJ"], SV["LABJ"], SV["RPMJ"]
    por_rpm, jan_ok, jan_tot = {}, 0, 0
    for rpm, v in jn.items():
        a, n = trava[rpm]
        i_n = [i for i in range(12) if LAB[i] == "normal" and RPM[i] == int(rpm)][0]
        En = EJ[OJ[i_n]:OJ[i_n + 1]]
        mu, var, T = c.modelo(En, a, a + n)
        meio = len(En) // 2
        x = dict(trava=[int(a), int(n)], q_oos=v["q_oos"], corretas=int(v["corretas"]), T=float(T))
        for nome, X in [("normal", En[meio:])] + [(o, EJ[OJ[i]:OJ[i + 1]]) for o in ("inner", "outer", "ball")
                                                 for i in range(12) if LAB[i] == o and RPM[i] == int(rpm)]:
            kk = int((c.pontua(X, mu, var) > T).sum())
            ok_jan = (len(X) - kk) if nome == "normal" else kk
            jan_ok += ok_jan
            jan_tot += len(X)
            x[nome] = dict(k=kk, n=len(X), kmin=int(v[nome][2]), pct_marcadas=round(100.0 * kk / len(X), 2),
                           falha=bool(kk >= v[nome][2]), certa=bool(v[nome][3]), janelas_certas=int(ok_jan))
        por_rpm[rpm] = x
    res.update(por_rpm=por_rpm, corretas=int(sum(v["corretas"] for v in por_rpm.values())), total=12,
               janelas_certas=int(jan_ok), janelas_total=int(jan_tot), acc_janela_pct=round(100.0 * jan_ok / jan_tot, 2))
    return res


# ------------------------------------------------------------------ MAFAULDA
def metricas(R, col):
    nor, op = R[R.grupo == "normal"], R[R.grupo == "open-set"]
    tp, fp = int(op[col].sum()), int(nor[col].sum())
    fn, tn = len(op) - tp, len(nor) - fp
    rec = tp / (tp + fn)
    esp = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp + fp else float("nan")
    x, y = (op.k / op.n).values, (nor.k / nor.n).values
    rk = rankdata(np.concatenate([x, y]))
    auc = (rk[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
    fam = {cl: [int(d[col].sum()), int(len(d))] for cl, d in op.groupby("classe")}
    db = R[R.grupo == "desbal"]
    ov = R[R.grupo == "open-set V"]
    return dict(tp=tp, fp=fp, fn=int(fn), tn=int(tn), acuracia=(tp + tn) / (tp + tn + fp + fn), recall=rec,
                especificidade=esp, precisao=nan_none(prec), f1=nan_none(f1), gmean=math.sqrt(rec * esp), auc_Pr=float(auc),
                por_familia=fam, desbal=[int(db[col].sum()), int(len(db))], open_set_V=[int(ov[col].sum()), int(len(ov))],
                n_aberto=int(len(op)), n_normal=int(len(nor)))


def bloco_maf(tag):
    x = ap.mafaulda(tag)
    if x is None:
        return None
    res, R = x
    return dict(resumo=res, loos=metricas(R, "alarme_loos"), artigo=metricas(R, "alarme_artigo"),
                normais_alarmadas_loos=[os.path.basename(str(r)) for r in R[(R.grupo == "normal") & R.alarme_loos].relpath],
                normais_alarmadas_artigo=[os.path.basename(str(r)) for r in R[(R.grupo == "normal") & R.alarme_artigo].relpath])


# ------------------------------------------------------------------ uma etiqueta
def roda_tag(tag):
    t0 = time.time()
    pasta = ap.pasta_v10(tag)
    if not (pasta.startswith(ap.PREFIXO_ZIP) or os.path.isdir(pasta)):
        diz("%s: %s nao existe, pulando" % (tag, pasta))
        return None
    diz("=" * 90)
    diz("%s | pasta %s" % (tag, pasta))
    SV = c.ambiente(pasta)
    res = dict(tag=tag, pasta=ap.relativo(pasta))
    res["ims"] = bloco_ims(SV)
    diz("   IMS pronto (%.0f s)" % (time.time() - t0))
    res["grade"] = bloco_grade(SV, res["ims"])
    for r in res["ims"].values():
        r.pop("_S", None)
    diz("   grade pronta: %d (%d) de %d (%.0f s)" % (res["grade"]["tres_de_tres"], res["grade"]["tres_pelo_eixo"],
                                                  res["grade"]["n_celulas"], time.time() - t0))
    res["pron"] = bloco_pron(SV)
    res["jnu"] = bloco_jnu(SV)
    diz("   PRONOSTIA %d/17, JNU %s (%.0f s)" % (res["pron"]["alarmam"], res["jnu"].get("corretas", "sem trava"), time.time() - t0))
    SV["CACHE"].clear()
    SV["FLAGS"].clear()
    del SV
    gc.collect()
    res["mafaulda"] = bloco_maf(tag)
    gc.collect()
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "resultado_%s.json" % tag), "w", encoding="utf-8"), indent=1, default=conv)
    diz(resumo_tag(res))
    return res


# ------------------------------------------------------------------ texto
def resumo_tag(r):
    L = []
    cel = []
    for seq in ("1st_test", "2nd_test", "4th_test"):
        x = r["ims"][seq]
        if x["cat"] == "recusa":
            cel.append("%s: recusa (%s)" % (NOME[seq], x["motivo"]))
            continue
        d = x["desfecho"]
        base = "%s: ref %g+%g h (fim dia %.2f)" % (NOME[seq], x["s0"] / 6, x["n"] / 6, x["fim_com_dia"])
        if "lead_d" in d:
            cel.append("%s, %s rol %d a %.2f d" % (base, x["cat"], d["rol"], d["lead_d"]))
        else:
            cel.append("%s, %s" % (base, d["cat"]))
    cert = sum(r["ims"][s].get("conta_acerto", False) for s in r["ims"])
    L.append("%s | IMS %d/3 | %s" % (r["tag"], cert, " | ".join(cel)))
    g = r["grade"]
    L.append("   grade s0 x n: %d (%d) de %d; certas por set %s" % (g["tres_de_tres"], g["tres_pelo_eixo"], g["n_celulas"], g["certas_por_set"]))
    custo = []
    for seq in r["ims"]:
        x = r["ims"][seq]
        if "rol" in x:
            custo.append("%s: %s" % (NOME[seq], " ".join("r%s=%d%s" % (b, len(v["linha_do_tempo"]), "F" if v["falhou"] else "")
                                                         for b, v in x["rol"].items())))
    L.append("   alarmes por rolamento (custo em inspecoes): " + " | ".join(custo))
    p = r["pron"]
    L.append("   PRONOSTIA %d/%d (sem alarme: %s); mediana Rem %.0f %%, lead %.2f h; nos 10 min: %d"
             % (p["alarmam"], p["total"], p["sem_alarme"], p["mediana_rem"] or float("nan"), p["mediana_lead_h"] or float("nan"), p["nos_10min"]))
    j = r["jnu"]
    if "corretas" in j:
        L.append("   JNU %d/12; janelas certas %d/%d (%.2f %%); trava %s" % (j["corretas"], j["janelas_certas"], j["janelas_total"], j["acc_janela_pct"], j["trava"]))
    else:
        L.append("   JNU sem trava: %s" % j["motivo"])
    m = r["mafaulda"]
    if m:
        L.append("   MAFAULDA Loos %d/%d (normais %d); artigo %d (normais %d); AUC(P_r) %.4f"
                 % (m["loos"]["tp"], m["loos"]["n_aberto"], m["loos"]["fp"], m["artigo"]["tp"], m["artigo"]["fp"], m["loos"]["auc_Pr"]))
    else:
        L.append("   MAFAULDA: sem embeddings")
    return "\n".join(L)


def junta():
    tudo = {}
    for tag in TAGS:
        p = os.path.join(OUT, "resultado_%s.json" % tag)
        if os.path.exists(p):
            tudo[tag] = json.load(open(p, encoding="utf-8"))
    json.dump(tudo, open(os.path.join(OUT, "numeros.json"), "w", encoding="utf-8"), indent=1, default=conv)
    io.open(os.path.join(OUT, "numeros.txt"), "w", encoding="utf-8").write("\n\n".join(resumo_tag(r) for r in tudo.values()))
    diz("numeros.json e numeros.txt: %d etiquetas (%s)" % (len(tudo), ", ".join(tudo)))


if __name__ == "__main__":
    refaz = "--refaz" in ARGS
    tags = [a for a in ARGS if not a.startswith("--")] or TAGS
    for tag in tags:
        if os.path.exists(os.path.join(OUT, "resultado_%s.json" % tag)) and not refaz:
            diz("%s: ja existe, pulando" % tag)
            continue
        roda_tag(tag)
        gc.collect()
    junta()
