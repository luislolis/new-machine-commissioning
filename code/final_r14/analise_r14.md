## 5. Resultados

Gerado por `code/analise_r14.py`. Os critérios são os da Seção 3, declarados antes de rodar.

**Fidelidade do extrator:** r9 do `rerun_extract14` contra `code/rerun9`, 30 arquivos, maior diferença 1.22e-04.

**Referência:** as seis rodadas de treino com desbalanceamento.

| Treino | MAFAULDA (normais marcadas) | Set 1 | Set 2 | Set 3 | Grade 3/3 | PRONOSTIA (10 min) | JNU (por janela) |
|---|---|---|---|---|---|---|---|
| r9 | 1500/1538 (0) | acerto r4, 5.17 | acerto r1, 3.06 | longe r2, 42.98 | 24 | 16/17 (2) | 12/12 (87.6 %) |
| d0 | 1516/1538 (0) | acerto r4, 5.37 | acerto r1, 2.97 | longe r2, 42.92 | 17 | 17/17 (3) | 12/12 (90.8 %) |
| d1 | 1537/1538 (0) | longe r1, 22.57 | acerto r1, 3.13 | longe r2, 43.78 | 3 | 15/17 (1) | 12/12 (87.5 %) |
| d2 | 1528/1538 (0) | suspeito r4, 24.84 | acerto r1, 3.10 | longe r2, 42.47 | 0 | 17/17 (3) | 12/12 (83.7 %) |
| d3 | 1510/1538 (0) | longe r1, 27.28 | acerto r1, 2.98 | longe r2, 42.81 | 15 | 17/17 (3) | 12/12 (89.5 %) |
| d4 | 1528/1538 (0) | suspeito r4, 22.57 | acerto r1, 2.61 | longe r2, 42.98 | 0 | 17/17 (2) | 12/12 (81.6 %) |

**Rodada 14:** todas as famílias. No MAFAULDA, as falhas foram vistas no treino; só os blocos normais marcados valem como teste.

| Treino | MAFAULDA (normais marcadas) | Set 1 | Set 2 | Set 3 | Grade 3/3 | PRONOSTIA (10 min) | JNU (por janela) |
|---|---|---|---|---|---|---|---|
| r14b0 | 1537/1538 vistas (0) | longe r1, 27.14 | acerto r1, 3.03 | longe r2, 43.04 | 7 | 17/17 (3) | 12/12 (91.2 %) |
| r14b1 | 1538/1538 vistas (0) | longe r1, 27.28 | acerto r1, 3.05 | longe r2, 43.07 | 5 | 17/17 (4) | 12/12 (90.5 %) |
| r14b2 | 1538/1538 vistas (0) | suspeito r4, 18.35 | acerto r1, 2.62 | longe r2, 42.93 | 0 | 17/17 (2) | 12/12 (88.5 %) |
| r14m0 | 1538/1538 vistas (0) | longe r1, 22.44 | acerto r1, 3.06 | longe r2, 42.44 | 4 | 17/17 (3) | 12/12 (90.0 %) |
| r14m1 | 1538/1538 vistas (0) | eixo r1, 27.34 | acerto r1, 3.06 | longe r2, 42.42 | 0 | 17/17 (4) | 12/12 (85.0 %) |
| r14m2 | 1538/1538 vistas (0) | longe r1, 27.38 | acerto r1, 2.71 | longe r2, 37.55 | 4 | 17/17 (3) | 12/12 (91.9 %) |

**Veredito de H1:**
- binario: **não passa**. Set 1 acerta em 0 de 3; falha em: set1.
- multiclasse: **não passa**. Set 1 acerta em 0 de 3; falha em: set1.
- Previsão declarada para o Set 3 (alarme longe nas seis): confirmada.

**R² da forma do espectro sobre o embedding** (descritivo; no artigo o r9 dá 0,61): r9 0.61, r14b0 0.76, r14b1 0.72, r14b2 0.74, r14m0 0.75, r14m1 0.72, r14m2 0.74.

**AUCs da cadeia** (descritivo): ver `code/final_r13/auc_cadeia.txt`.
