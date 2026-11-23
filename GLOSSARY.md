# Glossary: evaluation levels, metric keys and sets

## Gesture classes

| Class | Meaning |
|---|---|
| G1 | Open the medication container |
| G2 | Pick up the pill |
| G3 | Pill to mouth |
| G4 | Drink water |
| G5 | Rest / any other movement (daily-life negatives) |

An intake is a session in which the pill reaches the mouth (G3).

## Evaluation levels

| Level | What is compared | Example |
|---|---|---|
| **Window** | The class predicted for each window against the window's label (the label of its centre sample, as in training). Raw classifier output, no temporal reconstruction. | window 17 → G2 (true G2) |
| **Gesture Timeline** | The label at every time step of the session on a common 10 Hz frame grid (0.1 s). Built from the window predictions (each frame averages the probabilities of the windows covering it), optionally decoded by an HMM or HSMM. | `G1 G1 G1 G2 G2 G3 G3 G4 G4 G4` |
| **Gesture Order** | The Gesture Timeline with runs of the same label collapsed and G5 removed. Only the order of the gestures matters. | `G1 → G2 → G3 → G4` |

The common frame grid makes configurations with different window lengths and overlaps
comparable: a 0.5 s / 75 % model and a 5 s / 0 % model produce very different numbers of windows
but the same number of timeline frames.

## Metrics

Keys are `<set>.<level>.<metric>` in `index/runs.csv` and `index/postproc.csv`, and
`<set> → <level> → <metric>` in every `metrics.json`.

| Level | Metric | Key | Definition |
|---|---|---|---|
| Window | Precision, recall, F1 per class | `window.precision.<G>`, `window.recall.<G>`, `window.f1.<G>` | TP / (TP + FP), TP / (TP + FN), their harmonic mean |
| Window | Macro F1 | `window.macro_f1` | mean of the per-class F1 over the classes present |
| Window | Accuracy | `window.accuracy` | correct windows / all labelled windows |
| Window | Confusion matrix | `window.confusion` | 5 × 5, rows = true, columns = predicted |
| Timeline | The same per frame | `timeline.*` | as for windows, on the 10 Hz frames |
| Timeline | **Segmental F1@k** | `timeline.seg_f1_10`, `timeline.seg_f1_25`, `timeline.seg_f1_50` | Lea et al. (2017): a predicted gesture segment is a TP if it has the right class and IoU ≥ k with a still unmatched true segment; TP / FP / FN pooled over the set |
| Order | **Order Exact Match** | `order.exact_match` | fraction of sessions whose Gesture Order equals the annotated one |
| Order | Order Edit Distance | `order.edit_distance` | Levenshtein distance between predicted and true Gesture Order, divided by the true length |
| Intake | Sensitivity, specificity, balanced accuracy | `intake_g3.*`, `intake_g2_g3.*`, `intake_llr_fa05.*`, `intake_llr_fa10.*` | per session, intake decided vs. annotated. `g3`: a G3 segment in the timeline; `g2_g3`: G2 followed later by G3; `llr_faXX`: phased HMM log-likelihood ratio above a threshold calibrated for ≤ 0.5 (`fa05`) or ≤ 1.0 (`fa10`) false events per hour |
| External | Events per hour | `external.gesture_events_per_hour`, `external.g3_events_per_hour`, `external.llr_fa05_events_per_hour`, `external.llr_fa10_events_per_hour` | events on the held-out InGesture / WISDM participants, who perform no medication gestures |

## Sets

| Set (file name) | Content | Use |
|---|---|---|
| `val` | held-out gesture subjects of the fold composed into daily activity of the external calibration participants (worlds W3int and W4long) | early stopping, every selection, decoder tuning |
| `test_sequences` | the annotated lab sessions (`data/labeled_output`, different people from the gesture trials) | blind test |
| `test_wild` | the same sessions embedded in daily activity of the held-out external participants | blind test |
| `test_external` | held-out InGesture / WISDM participants alone | false events per hour |
| `external_calibration` | InGesture / WISDM calibration participants | intake thresholds under the false-event budget |

## Selection criterion

Every "best model" choice uses **`val.timeline.seg_f1_25`** (mean over the replicates of a
configuration), with `val.window.macro_f1` (`val.timeline.macro_f1` in E3) as the tie-breaker.
The blind sets never drive a choice.
