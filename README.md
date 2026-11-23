# Wrist-IMU gesture recognition and medication-intake detection

Code to reproduce three studies from the annotated smartwatch data in `data/` and two public
datasets:

1. **Experiments E1–E3.** Window classifiers for five medication-intake gestures (E1: trained on
   the annotated gestures only; E2: with daily-life negatives from InGesture and WISDM), selected
   by a staged sweep, then temporal decoding (HMM / HSMM) and an intake decision (E3).
2. **Decoder study (D0–D5).** Temporal decoders compared on the frozen E1/E2 classifiers: none,
   mode filter, HMM, HSMM, hidden-state CRF variants and a learned refiner, offline and with a fixed
   lag, with participant-level statistics.
3. **Embedded study.** Where the pipeline should run (phone, watch or split). Phase 0, on the PC:
   causal filtering, retraining, ONNX precisions, fixed-lag Viterbi and streaming LLR. Phase 1:
   the Android apps and the device benchmark. Phase 2: the energy campaign.

Terms, metric keys and sets are defined in [`GLOSSARY.md`](GLOSSARY.md).

## Layout

```
├── conf/                  configuration (JSON)
│   ├── paths.json         data and output locations, relative to the repository root
│   ├── experiment_plan.json   E1–E3: factors, grids, stages, selection rule
│   ├── training.json      training recipe: data, composition, augmentation, optimiser, HMM settings
│   ├── worlds.json        composed training worlds, training mix per experiment, filters, default hyperparameters
│   └── embedded.json      embedded study: finalist, replicates, filter variants, campaign settings
├── data/
│   ├── labeled_output_gestures/   isolated gesture trials, 15 subjects (training and validation)
│   ├── labeled_output/            lab protocol sessions, 10 other subjects (blind test)
│   └── third_party/               InGesture and WISDM (scripts/download_datasets.py)
├── src/
│   ├── core/              data loading, stream composition, models, training, HMM, HSMM
│   ├── thesis/            E1–E3: training, sweep, metrics, post-processing, registry, reports
│   ├── decoders/          decoder study
│   └── embedded/          embedded study (PC side and device control)
├── android/               Kotlin sources of the watch and phone apps (core, benchlib, wear, phone)
├── scripts/               setup, dataset download and the run scripts of each study
├── tests/                 unit tests (pytest)
└── outputs/
    ├── thesis/ decoders/ embedded/   written by the scripts
    └── reference/                    final models and report tables of the reported runs
```

## Setup

Python 3.10 or newer. A CUDA GPU is needed for the training stages in practice; everything else
runs on the CPU.

```
bash scripts/setup_env.sh              # creates .venv with requirements.txt
.venv/bin/python scripts/download_datasets.py
.venv/bin/python -m pytest             # unit tests
```

`download_datasets.py` downloads, verifies (SHA-256) and extracts:

| Dataset | Source | License | Target |
|---|---|---|---|
| InGesture v5 | Mendeley Data, [doi:10.17632/fdxst56tcj.5](https://doi.org/10.17632/fdxst56tcj.5) | CC BY 4.0 | `data/third_party/InGesture/` |
| WISDM 2019 | UCI ML Repository, [doi:10.24432/C5HK59](https://doi.org/10.24432/C5HK59) | CC BY 4.0 | `data/third_party/WISDM2019/` |

Mendeley Data does not always serve its archive to scripts. If the InGesture download fails,
download `fdxst56tcj-5.zip` from the dataset page ("Download All") and run
`python scripts/download_datasets.py ingesture --archive <path>`.

## Data

* `data/labeled_output_gestures/<Subject>/<G1..G5>/*.csv`: one isolated gesture per file, 100 Hz,
  columns `acc_x acc_y acc_z` (m/s²) and `gyr_x gyr_y gyr_z` (deg/s).
  `participant_summary.csv` and `split_manifest.json` describe the files.
* `data/labeled_output/<Subject>/labeled_<subject>_<time>_<intake>_<protocol>.csv`: continuous lab
  sessions, 100 Hz, the same sensor columns and the per-sample label `Gesto` (G1–G5). `intake` is
  0/1; the protocols S01–S05 are listed in `src/core/sources.py`, S02M is G3 → G4.

Every signal is resampled to the configuration's rate (anti-alias low-pass), converted to g and
deg/s and, unless the filter factor is `none`, high-pass filtered at 0.3 Hz on the accelerometer.
Training streams are composed from the gesture trials of the fold's training subjects
(`src/core/streams.py`); the lab sessions are only ever tested.

## Running

Every command runs from the repository root. The scripts use `.venv/bin/python` (set `PY` to use
another interpreter) and put `src/` on `PYTHONPATH`; individual steps run as
`python -m <package>.<module>` with `PYTHONPATH=src`. Every step resumes where it stopped.

| Script | What it runs | Needs |
|---|---|---|
| `scripts/run_thesis.sh` | E1 and E2 sweep (stages A → B → C → F), short-window tracks, E3, model registry, reports | data, GPU |
| `scripts/run_decoders.sh` | decoder study D0 → D5 | finished `run_thesis.sh` (it reads the saved predictions of every replicate) |
| `scripts/run_embedded.sh` | embedded study phase 0 and the PC part of phase 1 | finished `run_thesis.sh`, GPU for the causal retraining |
| `scripts/run_all.sh` | the three above in order | |

`WORKERS` sets the number of CPU processes of the decoding steps. `python -m thesis.sweep --capacity N`
sets how many training runs share the GPU at once (default 2, for a 4 GB GPU), and
`python -m thesis.sweep --dry-run` lists the work.

### Experiments E1–E3 (`src/thesis/`)

| Module | Role |
|---|---|
| `common` | output paths, config keys, hyperparameter grid, training config of a configuration |
| `train_config` | trains one configuration (folds × seeds) and writes its run directory |
| `metrics` | every metric from saved predictions (`--force` recomputes) |
| `sweep` | stages A → B → C → F for E1 and E2, selection files, `index/runs.csv` |
| `extra_windows` | short-window tracks (best 0.75 s and 1 s of E1 and E2) through B → C → F |
| `postproc` | E3: HMM / HSMM grid on validation, selected decoders on every set, intake thresholds |
| `registry` | deployment training (`fall_s0`), ONNX export, model cards, `models/registry.csv` |
| `report` | `reports/<experiment>/REPORT.md`, tables and figures |
| `results_chapter` | `reports/results_chapter/`: E1 → E2 → comparison at window, sequence and intake level |

Stages (selection by mean `val.timeline.seg_f1_25`, tie-break `val.window.macro_f1`):

| Stage | Varies | Replicates |
|---|---|---|
| A, signal | fs × window × overlap × signals × filter (240 configurations, CNN-BiLSTM, default hyperparameters) | 3 folds × seed 0 |
| B, model | CNN / CNN-LSTM / CNN-BiLSTM on the 5 best signals of A | 3 folds × seeds 0–2 |
| C, hyperparameters | the 64-point grid on the 3 best of B | 3 folds × seed 0 |
| F, finalists | the 3 best of C | 3 folds × seeds 0–2 |

#### Config key

`<model>__fs<Hz>_w<window×100, 3 digits>_ov<overlap×100, 2 digits>_<signals>_<filter>__<hp>`,
for example `cnn_bilstm__fs100_w200_ov75_accgyr_hp03__hp05`.

| Part | Values |
|---|---|
| `model` | `cnn`, `cnn_lstm`, `cnn_bilstm` |
| `fs` | `fs20`, `fs50`, `fs100` |
| `w` | `w050`, `w075`, `w100`, `w200`, `w500` |
| `ov` | `ov00`, `ov25`, `ov50`, `ov75` |
| `signals` | `acc`, `accgyr` |
| `filter` | `hp03` (0.3 Hz high-pass on the accelerometer, zero-phase), `hp03c` (the same, causal), `raw` |
| `hp` | `hpdef` (default hyperparameters) or `hp00`…`hp63` (index into the grid, in key order) |

A replicate is `f<fold>_s<seed>`; the deployment model trained on every training subject is
`fall_s0`. Decoder keys are `<decoder>__<params>` (for example `hmm__ov75_st095_md030_t200`: every
window, rest self-transition 0.95 per second, minimum-duration fraction 0.3, emission temperature
2.0); `none` is the E1/E2 timeline rule.

#### Outputs (`outputs/thesis/`)

```
runs/<experiment_slug>/<config_key>/
├── config.json, log.txt, stdout.txt
├── artifacts/manifest.json        fs, window, stride, folds and subjects, every evaluation stream with its truth
├── artifacts/frames.npz           labels per stream on the window grid (y/), the 10 Hz grid (y10/), edge distances
└── <replicate>/
    ├── manifest.json              status (running / done / failed), subjects, timings
    ├── model.pt, preprocessing.json, train_info.json, train_log.csv
    ├── predictions/<set>.npz      window probabilities (float16) per stream
    └── metrics.json, sessions.csv
postproc/<experiment_slug>/<config_key>/<replicate>/
├── grid_val.csv                   every grid point on the validation streams
└── <decoder_key>/{metrics.json, sessions.csv, paths.npz, summary_row.json}
index/runs.csv, index/postproc.csv, index/selection/*.json
models/<name>/v<k>/{model.pt, model.onnx, preprocessing.json, postprocessing.json, metrics.json, model_card.json}
reports/
```

Window j of a stream is centred on sample `j × stride`, so the predictions and the manifest are
enough to rebuild any timeline without retraining.

### Decoder study (`src/decoders/`)

Reads only the saved predictions in `outputs/thesis/runs`, through a loader that refuses every set
except `val`, `test_wild`, `external_calibration` and `test_external` (the sets opened are logged in
`outputs/decoders/index/sets_read.txt`).

| Stage | Module | Content |
|---|---|---|
| D0 | `d0_reproduce`, `d0_report` | reproduces the E3 decoders on `test_wild` through the restricted loader (check against `index/postproc.csv`) and the participant-level baseline |
| D1 | `d1_run`, `d1b_run`, `d1_report` | every decoder on the E3 replicates (panel F), decoder-specific intake decisions |
| D2 | `d2_run`, `d2_report` | none / mode / HMM / HSMM on every replicate of the sweep (panel S): decoder gain vs. classifier quality |
| D3 | `d3_run`, `d3_report` | fixed-lag (online) decoding: accuracy vs. latency |
| A4, D5 | `a4_tuning_data`, `d5_report` | validation-data fraction needed by each decoder; summary of analyses A1–A6 |

`decoders` holds the decoders, `stats` the participant-level statistics (cluster bootstrap,
paired Wilcoxon tests with Holm correction). Outputs: `outputs/decoders/{index, reports}`.

### Embedded study (`src/embedded/`, `android/`)

The finalist is the best E2 configuration after E3 (`conf/embedded.json` can fix another
100 Hz configuration).

| Phase | Module | Content |
|---|---|---|
| 0 | `phase0_predict` | predictions of the finalist on streams with the zero-phase or the causal high-pass |
| 0 | `phase0_quant` | ONNX FP32 / FP16 / INT8 exports of the causally retrained model and their predictions |
| 0 | `phase0_eval`, `phase0_report` | variants, fixed-lag Viterbi and streaming-LLR latency |
| 1 | `devicepipe` | reference implementation of the on-device pipeline with streaming semantics |
| 1 | `phase1_export` | `android/bundle/` (pipeline.json, ONNX models), replay files, golden test fixtures |
| 1 | `phase1_reference` | the PC reference the devices must reproduce |
| 1–2 | `phase1_bench` | builds, installs, runs and analyses the benchmark on the devices (adb) |
| 2 | `phase2_campaign`, `phase2_analysis` | energy campaign (conditions C0, C1, P1–P3) and its cost / energy tables |

Phases 1 and 2 need a Wear OS watch, an Android phone, adb and the Android SDK:

```
python -m embedded.phase1_bench devices
python -m embedded.phase1_bench install
python -m embedded.phase1_bench push
python -m embedded.phase1_bench run --on watch --set condition=LOCAL pacing=fast steps=2,5,10 log_probs=true --wait
python -m embedded.phase1_bench analyze <run_id>
python -m embedded.phase2_campaign run --rep <k> --phone <serial>
python -m embedded.phase2_campaign pull-watch --rep <k> --watch <serial> --phone <serial>
python -m embedded.phase2_campaign summary --reps 1,2,...
python -m embedded.phase2_analysis
```

The Kotlin parity tests (`cd android && ./gradlew :core:test`, add `-Preplay` for the whole replay)
need the bundle, the golden fixtures and the replay files written by `embedded.phase1_export`
and `embedded.phase1_reference`; `android/local.properties` must point `sdk.dir` to the Android SDK.

## Reference outputs (`outputs/reference/`)

The results of the reported runs, for checking without retraining:

* `thesis/models/`: the six registered models (weights, ONNX, pre- and post-processing parameters,
  metrics, model cards) and `registry.csv`;
* `thesis/reports/`: E1, E2, E3, short-window and results-chapter tables and figures;
* `thesis/index/`: `runs.csv`, `postproc.csv` and every selection file;
* `decoders/reports/`: the reports of stages D0–D5;
* `embedded/`: the phase-0 tables and figures, the deployment bundle, the phase-1 PC reference
  summary, the phase-2 device measurements (`phase1/devices/e2_rep*`) and the energy analysis.

The saved window predictions of the sweep are not included; the decoder study regenerates them by
running `scripts/run_thesis.sh` first.

## Tests

`python -m pytest` runs the unit tests on synthetic data (sources, filters, composition, models,
training, HMM / HSMM, metrics, decoders, device pipeline, ONNX export, dataset script, command-line
entry points). Tests marked `data` and `third_party` check the files in `data/` and are skipped
when the data is absent. The tests write only to a temporary directory.
