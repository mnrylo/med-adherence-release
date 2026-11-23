# Stage D0: reproduction and participant-level baseline

Generated 2026-11-23 17:53:21. Test set: `test_wild` only (lab sessions interleaved with InGesture/WISDM).

## 1. Verification

| Check | Result | Note |
|---|---|---|
| V2 reproduction | PASS | 2700 values (90 replicates × 3 decoders × 10 metrics) against the E3 postproc.csv; max abs diff 2.2e-16 |
| V3 data isolation | PASS | sets opened: external_calibration.npz, test_wild.npz, val.npz |

## 2. Test participants

Sessions of `test_wild` per participant. Every participant weighs the same in the participant-level means below, so Subject15 (1 session) weighs as much as Subject01 (50); the session-pooled E3 numbers are shown next to them for comparison.

| Subject01 | Subject02 | Subject03 | Subject05 | Subject06 | Subject07 | Subject08 | Subject09 | Subject15 | Subject16 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 50 | 11 | 50 | 50 | 50 | 50 | 20 | 1 | 15 |

## 3. E2 finalists (2 s CNN-BiLSTM, acc+gyr)

3 configurations, 27 replicates. Participant-level mean [95 % cluster-bootstrap CI]; Δ = decoder − HMM [95 % CI], Holm-adjusted Wilcoxon p over participants.

| Metric | none | HMM | HSMM | Δ none | Δ HSMM |
|---|---|---|---|---|---|
| Seg F1@25 | 89.0 [86.1, 91.4] | 91.0 [88.2, 93.3] | 92.0 [89.0, 94.5] | -2.0 [-3.1, -0.5], p = 0.074 | +1.0 [+0.3, +1.7], p = 0.074 |
| Seg F1@10 | 89.4 [86.9, 91.6] | 91.2 [88.7, 93.4] | 92.3 [89.6, 94.6] | -1.9 [-3.0, -0.4], p = 0.074 | +1.0 [+0.3, +1.7], p = 0.074 |
| Seg F1@50 | 85.6 [81.9, 89.0] | 84.8 [78.6, 89.9] | 86.5 [80.9, 91.2] | +0.9 [-3.1, +7.4], p = 0.160 | +1.7 [+1.0, +2.5], p = 0.004 |
| Order Exact Match | 57.5 [42.5, 69.9] | 59.4 [40.6, 75.1] | 67.0 [49.0, 81.7] | -1.9 [-6.4, +4.4], p = 0.084 | +7.7 [+4.8, +10.5], p = 0.004 |
| G3-rule sens | 99.1 [97.6, 99.9] | 99.3 [97.9, 100.0] | 98.9 [97.3, 100.0] | -0.2 [-0.6, +0.1], p = 0.500 | -0.3 [-0.8, +0.0], p = 0.500 |
| G3-rule spec | 95.6 [91.7, 98.7] | 97.4 [94.3, 99.2] | 98.0 [95.1, 99.6] | -1.7 [-4.5, +0.0], p = 0.062 | +0.7 [+0.2, +1.3], p = 0.062 |
| LLR sens (≤1 FA/h) | 99.1 [97.8, 99.8] | 99.1 [97.8, 99.8] | 99.1 [97.8, 99.8] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |
| LLR spec (≤1 FA/h) | 97.5 [94.7, 99.3] | 97.5 [94.7, 99.3] | 97.5 [94.7, 99.3] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |

Session-pooled (as in E3): hmm: seg F1@25 92.0, Order EM 70.6; hsmm: seg F1@25 93.3, Order EM 77.8; none: seg F1@25 89.2, Order EM 65.0

## 4. E2 short-window refs (0.75 / 1 s)

2 configurations, 18 replicates. Participant-level mean [95 % cluster-bootstrap CI]; Δ = decoder − HMM [95 % CI], Holm-adjusted Wilcoxon p over participants.

| Metric | none | HMM | HSMM | Δ none | Δ HSMM |
|---|---|---|---|---|---|
| Seg F1@25 | 74.8 [72.1, 77.5] | 86.2 [81.2, 90.2] | 87.7 [84.0, 91.0] | -11.4 [-16.0, -5.2], p = 0.039 | +1.5 [+0.1, +3.2], p = 0.039 |
| Seg F1@10 | 76.2 [73.8, 78.6] | 86.6 [81.5, 90.5] | 88.1 [84.3, 91.3] | -10.5 [-15.0, -4.4], p = 0.039 | +1.4 [+0.1, +3.1], p = 0.049 |
| Seg F1@50 | 63.3 [59.1, 67.3] | 78.8 [72.1, 84.9] | 81.0 [75.1, 86.5] | -15.5 [-20.3, -10.2], p = 0.004 | +2.2 [+0.9, +3.6], p = 0.004 |
| Order Exact Match | 26.8 [16.0, 37.4] | 48.8 [32.6, 62.5] | 52.8 [34.7, 68.2] | -21.9 [-32.7, -11.7], p = 0.008 | +4.0 [+0.7, +7.2], p = 0.051 |
| G3-rule sens | 99.8 [99.0, 100.0] | 95.7 [89.8, 99.6] | 95.7 [89.9, 99.6] | +4.1 [+0.3, +9.9], p = 0.250 | +0.0 [-0.5, +0.5], p = 1.000 |
| G3-rule spec | 67.3 [51.5, 80.6] | 94.0 [89.2, 97.7] | 93.9 [88.9, 97.6] | -26.7 [-40.6, -15.4], p = 0.031 | -0.0 [-5.3, +3.7], p = 0.578 |
| LLR sens (≤1 FA/h) | 98.2 [93.7, 99.9] | 98.2 [93.7, 99.9] | 98.2 [93.7, 99.9] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |
| LLR spec (≤1 FA/h) | 94.5 [89.0, 98.4] | 94.5 [89.0, 98.4] | 94.5 [89.0, 98.4] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |

Session-pooled (as in E3): hmm: seg F1@25 89.2, Order EM 58.1; hsmm: seg F1@25 89.9, Order EM 63.2; none: seg F1@25 74.6, Order EM 31.5

## 5. E1 finalists (5 s, gestures only)

3 configurations, 27 replicates. Participant-level mean [95 % cluster-bootstrap CI]; Δ = decoder − HMM [95 % CI], Holm-adjusted Wilcoxon p over participants.

| Metric | none | HMM | HSMM | Δ none | Δ HSMM |
|---|---|---|---|---|---|
| Seg F1@25 | 53.8 [50.8, 56.7] | 62.5 [59.2, 65.5] | 61.9 [58.7, 64.9] | -8.7 [-9.8, -7.5], p = 0.004 | -0.6 [-1.0, -0.3], p = 0.004 |
| Seg F1@10 | 55.8 [53.0, 58.5] | 64.8 [61.8, 67.5] | 64.0 [61.1, 66.7] | -9.0 [-10.2, -7.6], p = 0.004 | -0.9 [-1.2, -0.5], p = 0.004 |
| Seg F1@50 | 44.4 [39.9, 48.7] | 51.4 [46.5, 56.3] | 51.4 [46.6, 56.3] | -7.0 [-9.0, -4.2], p = 0.008 | +0.0 [-0.3, +0.4], p = 0.770 |
| Order Exact Match | 9.0 [6.7, 11.6] | 15.4 [11.5, 19.1] | 14.7 [10.9, 18.2] | -6.5 [-9.1, -4.0], p = 0.008 | -0.8 [-1.3, -0.2], p = 0.008 |
| G3-rule sens | 95.4 [91.8, 97.9] | 90.9 [86.1, 94.7] | 89.7 [85.0, 93.7] | +4.5 [+2.8, +6.5], p = 0.004 | -1.1 [-2.1, -0.3], p = 0.055 |
| G3-rule spec | 45.2 [36.2, 54.3] | 59.2 [50.9, 67.1] | 58.6 [50.2, 66.7] | -13.9 [-16.6, -11.5], p = 0.031 | -0.5 [-1.7, +0.5], p = 0.219 |
| LLR sens (≤1 FA/h) | 48.1 [38.6, 59.0] | 48.1 [38.6, 59.0] | 48.1 [38.6, 59.0] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |
| LLR spec (≤1 FA/h) | 95.7 [92.8, 97.7] | 95.7 [92.8, 97.7] | 95.7 [92.8, 97.7] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |

Session-pooled (as in E3): hmm: seg F1@25 62.9, Order EM 17.2; hsmm: seg F1@25 62.5, Order EM 16.4; none: seg F1@25 53.4, Order EM 9.4

## 6. E1 short-window refs

2 configurations, 18 replicates. Participant-level mean [95 % cluster-bootstrap CI]; Δ = decoder − HMM [95 % CI], Holm-adjusted Wilcoxon p over participants.

| Metric | none | HMM | HSMM | Δ none | Δ HSMM |
|---|---|---|---|---|---|
| Seg F1@25 | 33.7 [30.1, 38.5] | 56.4 [52.8, 60.0] | 52.2 [49.1, 55.6] | -22.7 [-26.6, -17.3], p = 0.004 | -4.2 [-5.2, -3.2], p = 0.004 |
| Seg F1@10 | 34.7 [31.2, 39.2] | 58.0 [54.4, 61.6] | 53.6 [50.3, 56.9] | -23.3 [-27.2, -17.9], p = 0.004 | -4.5 [-5.5, -3.5], p = 0.004 |
| Seg F1@50 | 28.7 [25.2, 33.1] | 50.2 [46.0, 54.1] | 46.9 [43.3, 50.6] | -21.5 [-24.8, -17.4], p = 0.004 | -3.2 [-4.2, -2.2], p = 0.004 |
| Order Exact Match | 1.2 [0.2, 2.6] | 7.1 [3.7, 10.9] | 5.8 [2.9, 9.1] | -5.9 [-8.9, -3.2], p = 0.008 | -1.3 [-2.2, -0.5], p = 0.008 |
| G3-rule sens | 100.0 [100.0, 100.0] | 98.7 [96.7, 99.8] | 99.2 [97.6, 99.9] | +1.3 [+0.2, +3.3], p = 0.062 | +0.5 [+0.0, +1.3], p = 0.062 |
| G3-rule spec | 5.4 [1.4, 11.1] | 21.1 [13.2, 30.0] | 18.3 [11.1, 26.4] | -15.7 [-22.3, -10.0], p = 0.031 | -2.8 [-5.1, -1.2], p = 0.031 |
| LLR sens (≤1 FA/h) | 12.9 [6.7, 22.2] | 12.9 [6.7, 22.2] | 12.9 [6.7, 22.2] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |
| LLR spec (≤1 FA/h) | 97.4 [94.6, 99.2] | 97.4 [94.6, 99.2] | 97.4 [94.6, 99.2] | +0.0 [+0.0, +0.0], p = 1.000 | +0.0 [+0.0, +0.0], p = 1.000 |

Session-pooled (as in E3): hmm: seg F1@25 57.4, Order EM 7.8; hsmm: seg F1@25 52.9, Order EM 6.4; none: seg F1@25 31.6, Order EM 1.2

## Notes

- Specificity uses only the participants with sessions without intake (7 of 10).
- The LLR decision is the E3 one: computed with the configuration's HMM for every decoder, so its columns do not depend on the decoder. Decoder-specific intake decisions come in stage D1.
- With 10 participants the smallest attainable two-sided Wilcoxon p is 0.002, before Holm correction.
