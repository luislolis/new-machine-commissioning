# A declared commissioning rule for vibration-based fault detection on new machines

Code, checkpoints and result files for the article

> M. S. Souza, L. H. A. Lolis, A. A. P. Sardá. *A Declared Commissioning Rule for Vibration-Based
> Fault Detection on New Machines: Evaluation on Four Datasets.* Submitted to Mechanical Systems and
> Signal Processing, 2026.

The derived data (window-level embeddings of the four datasets for every representation in the
article) are archived on Zenodo: **DOI [to be added]**. The code in this repository is the code
that produced the numbers, tables and figures of the article; comments and variable names are in
Portuguese.

## What the code does

The detector set-up of the article, the *commissioning rule*, is applied with the same constants on
four public datasets (MAFAULDA, IMS, PRONOSTIA, JNU):

- 1024-sample windows, z-scored per window;
- a 1-D CNN (`EmbeddingNet1D`, 6 x 1024 -> 128, L2-normalised), trained once on MAFAULDA with a
  triplet loss and then frozen (checkpoint `checkpoints/embedding_model_r9_seed42.pth`);
- a diagonal Mahalanobis model and a threshold T = Q0.99 estimated on the healthy reference of
  the unit;
- the reference is a 4-hour run test (24 acquisitions) for a nominal installation (IMS), the first
  3 s for an accelerated test (PRONOSTIA), and the part of the recording reserved for
  commissioning for single captures (JNU, MAFAULDA);
- a per-acquisition decision level k_min from a binomial model and a declared false-alarm target,
  with the out-of-sample rate measured between the two halves of the reference;
- an alarm after K = 7 consecutive anomalous acquisitions, and re-commissioning at documented
  stops, accepted when the broadband RMS stays within 1.5 dB of the commissioning reference.

## Repository layout

```
code/                        analysis scripts (run from this folder)
  final_v10.py               the rule on every dataset, for one or more representations (tags)
  figuras_v10.py             the result figures of the article
  figs_n0.py                 sensitivity of the IMS outcome to K (Section 3.3.4)
  diag_rnd_spec.py           R^2 of each embedding on the band-spectrum shape (Section 5.3)
  analise_r14.py             the trainings on all MAFAULDA fault categories and their criterion
  auc_r13.py                 window- and recording-level AUC of the chain
  deriva_r12.py              derives the level-free representations specn and featn from spec and feat
  apoio_v10.py, comissionamento_n0.py, regra_rms.py, nivel_oos.py,
  analise_final_v4.py, analise_sensibilidade_v4.py
                             the rule itself (reference, model, level, persistence, re-commissioning)
  analise_rotacoes_mafaulda.py
                             assigns each MAFAULDA recording to its nominal operating point
  mafaulda_rotacoes/         output of the script above (gravacoes.csv), used by training and analysis
  train_rodada*.py/.ipynb    training notebooks (Google Colab, GPU)
  rerun_extract*.py/.ipynb   embedding extraction for IMS, PRONOSTIA and JNU (Google Colab)
  final_v10/                 published results: resultado_<tag>.json, numeros.json, numeros.txt
  final_r13/, final_r14/, final_r10r11/, final_n0/
                             published results of the auxiliary analyses
checkpoints/                 the trained (and untrained) networks of Table 8, as PyTorch state dicts
figs/                        figures are written here
```

## Representations (tags)

| Tag | Representation | Training notebook | Extraction notebook |
|---|---|---|---|
| `r9` | reported checkpoint (seed 42) | `train_rodada9` | `rerun_extract9` |
| `d0`-`d4` | five further trainings (seeds 42-46, rotated blocks) | `train_rodada10` | `rerun_extract10` |
| `rnd0`, `rnd1` | the same network, untrained (two seeds) | `train_rodada11` | `rerun_extract11` |
| `zer` | reported weights, zero-filled input | - | `rerun_extract11` |
| `ce` | the same network trained as a cross-entropy classifier | `train_rodada11` | `rerun_extract11` |
| `spec`, `specn` | band spectrum, with level and level-free shape | - | `rerun_extract11`, `deriva_r12.py` |
| `feat`, `featn` | statistical indicators, with level and level-free shape | - | `rerun_extract11`, `deriva_r12.py` |
| `r14b0`-`r14b2` | triplets from all five MAFAULDA fault categories, binary objective (seeds 42-44) | `train_rodada14` | `rerun_extract14` |
| `r14m0`-`r14m2` | the same, multiclass objective (seeds 42-44) | `train_rodada14` | `rerun_extract14` |

`train_rodada9`, `train_rodada10`, `train_rodada11` and `train_rodada14` reuse sections of
`train_rodada7.py` at run time; keep it in the same Drive folder.

## Requirements

Analysis (CPU only): Python 3.11 and the versions in `requirements.txt`. Training and extraction
ran on Google Colab with PyTorch 2.x and read the raw datasets from Google Drive and Kaggle.

```
pip install -r requirements.txt
```

## Reproducing the article from the archived embeddings

1. Download the archives from Zenodo and place them as follows (paths relative to `code/`):

   | Archive | Where |
   |---|---|
   | `data_r9.zip` | extract into `code/` (creates `rerun9/`, `rodada9/`, `rerun4_artifacts/`) |
   | `rerun10_d0.zip` ... `rerun10_d4.zip`, `rodada10_treino.zip` | copy into `code/` **without** extracting (read in place) |
   | `data_alternative_external.zip`, `data_alternative_mafaulda.zip` | extract into `code/` (creates `rerun11/`, `rodada11/`) |
   | `rerun14_r14b0.zip` ... `rerun14_r14m2.zip`, `rerun14_r9.zip` | extract into `code/rerun14/` (`r9` there only checks that the extractor reproduces `rerun9/`) |
   | `rodada14_treino.zip` | extract into `code/rodada14/` |

   To keep the data elsewhere, set the environment variable `MISAEL_DADOS` to that folder for
   `rerun11/`, `rodada11/`, `rerun14/` and `rodada14/`; the others stay next to the scripts.

2. Run, from `code/`:

   ```
   python final_v10.py --refaz            # all 20 rows of Table 8, about 2.5 min per tag
   python figuras_v10.py ims first pronall jnu maf
   python figs_n0.py                      # K = 1 to 20 on IMS with the reference of the rule
   python diag_rnd_spec.py                # R^2 on the band-spectrum shape
   python analise_r14.py                  # criterion of the all-category trainings, with R^2
   ```

   `final_v10.py <tag> ...` runs selected tags; without `--refaz` it skips tags whose
   `final_v10/resultado_<tag>.json` already exists. The committed result files are the ones in the
   article, so `git diff code/final_v10` after a run shows any difference; only the run time
   (`segundos`) is expected to change.

| Article | Script | Output |
|---|---|---|
| Results on MAFAULDA, IMS, PRONOSTIA and JNU (Section 5), Table 8 | `final_v10.py` | `final_v10/resultado_<tag>.json`, `numeros.txt` |
| Result figures | `figuras_v10.py` | `figs/*_v10.png` |
| Grid of 100 references (Section 4.4) | `final_v10.py` (key `grade`) | `final_v10/resultado_<tag>.json` |
| Sensitivity to K (Section 3.3.4) | `figs_n0.py` | `final_n0/figs/` |
| R^2 on the spectral shape (Section 5.3) | `diag_rnd_spec.py`, `analise_r14.py` | `final_r10r11/diag_rnd_spec.json`, `final_r14/` |
| Trainings on all fault categories (Section 5.3) | `analise_r14.py` | `final_r14/analise_r14.md` |
| Hyperparameter grid (Section 4.3) | `train_rodada11` (Colab) | `hiper.csv` in `data_alternative_mafaulda.zip` |

## Raw datasets

The raw signals are not redistributed here. They are available from their original sources:

- MAFAULDA, Signals, Multimedia and Telecommunications Laboratory, Federal University of Rio de
  Janeiro: <https://www02.smt.ufrj.br/~offshore/mfs/page_01.html>
- IMS bearing data and FEMTO/PRONOSTIA (IEEE PHM 2012), NASA Prognostics Center of Excellence data
  repository: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- JNU bearing data: <https://github.com/ClarkGableWang/JNU-Bearing-Dataset>

## Licence

Code: MIT (see `LICENSE`). Checkpoints in `checkpoints/` and the data archived on Zenodo:
CC BY 4.0.

## Citation

See `CITATION.cff`. Until the article is published, please cite the Zenodo record.
