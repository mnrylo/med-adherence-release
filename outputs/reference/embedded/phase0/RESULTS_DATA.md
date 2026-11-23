# Phase 0 results (numbers)

Created 2026-11-23T17:53:46 from outputs/embedded/phase0/tables. Mean ± sd over the 9 fold replicates; LLR at the ≤ 1 FA/h budget (fa10).

## Variants (offline decoding)

| Variant | Thresholds | Window macro F1 (blind sequences) | Seg F1@25 (blind sequences) | Order EM (blind sequences) | Order EM (daily activity) | LLR sensitivity ≤1/h (daily activity) | LLR specificity (daily) | FA/h (test_external) |
|---|---|---|---|---|---|---|---|---|
| zero-phase (as trained) | recalibrated | 0.819 ± 0.008 | 0.950 ± 0.007 | 0.759 ± 0.036 | 0.729 ± 0.029 | 0.991 ± 0.005 | 0.977 ± 0.017 | 0.797 ± 0.320 |
| causal, not retrained | recalibrated | 0.658 ± 0.023 | 0.897 ± 0.020 | 0.580 ± 0.072 | 0.483 ± 0.064 | 0.884 ± 0.069 | 0.900 ± 0.068 | 0.518 ± 0.584 |
| causal, not retrained | from_zerophase | 0.658 ± 0.023 | 0.897 ± 0.020 | 0.580 ± 0.072 | 0.483 ± 0.064 | 0.852 ± 0.082 | 0.943 ± 0.047 | 0.321 ± 0.291 |
| causal, retrained | recalibrated | 0.811 ± 0.010 | 0.950 ± 0.013 | 0.772 ± 0.062 | 0.725 ± 0.052 | 0.966 ± 0.014 | 0.948 ± 0.015 | 0.942 ± 0.314 |
| retrained, ONNX FP16 | recalibrated | 0.811 ± 0.010 | 0.950 ± 0.013 | 0.771 ± 0.062 | 0.725 ± 0.051 | 0.966 ± 0.014 | 0.948 ± 0.015 | 0.942 ± 0.314 |
| retrained, ONNX FP16 | from_causal_trained | 0.811 ± 0.010 | 0.950 ± 0.013 | 0.771 ± 0.062 | 0.725 ± 0.051 | 0.966 ± 0.014 | 0.948 ± 0.015 | 0.942 ± 0.314 |
| retrained, ONNX FP32 | recalibrated | 0.811 ± 0.010 | 0.950 ± 0.013 | 0.772 ± 0.062 | 0.725 ± 0.052 | 0.966 ± 0.014 | 0.948 ± 0.015 | 0.932 ± 0.316 |
| retrained, ONNX FP32 | from_causal_trained | 0.811 ± 0.010 | 0.950 ± 0.013 | 0.772 ± 0.062 | 0.725 ± 0.052 | 0.967 ± 0.013 | 0.948 ± 0.015 | 0.942 ± 0.314 |
| retrained, ONNX INT8 dyn. | recalibrated | 0.811 ± 0.009 | 0.950 ± 0.014 | 0.774 ± 0.062 | 0.725 ± 0.051 | 0.968 ± 0.013 | 0.947 ± 0.013 | 0.921 ± 0.293 |
| retrained, ONNX INT8 dyn. | from_causal_trained | 0.811 ± 0.009 | 0.950 ± 0.014 | 0.774 ± 0.062 | 0.725 ± 0.051 | 0.969 ± 0.013 | 0.945 ± 0.018 | 0.963 ± 0.319 |
| retrained, ONNX INT8 dyn., batch 1 | recalibrated | 0.811 ± 0.009 | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.726 ± 0.053 | 0.967 ± 0.013 | 0.947 ± 0.019 | 0.952 ± 0.325 |
| retrained, ONNX INT8 dyn., batch 1 | from_causal_trained | 0.811 ± 0.009 | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.726 ± 0.053 | 0.967 ± 0.013 | 0.944 ± 0.017 | 1.004 ± 0.345 |

## Fixed-lag Viterbi (retrained, ONNX INT8 dyn., batch 1)

| Lag (s) | Record latency (s) | Seg F1@25 seq | Order EM seq | Seg F1@25 daily | Order EM daily |
|---|---|---|---|---|---|
| 0 | 1.5 | 0.906 ± 0.016 | 0.640 ± 0.047 | 0.866 ± 0.013 | 0.576 ± 0.052 |
| 0.5 | 2.0 | 0.916 ± 0.015 | 0.643 ± 0.047 | 0.880 ± 0.010 | 0.575 ± 0.052 |
| 1 | 2.5 | 0.916 ± 0.015 | 0.656 ± 0.055 | 0.878 ± 0.009 | 0.584 ± 0.049 |
| 2 | 3.5 | 0.932 ± 0.018 | 0.742 ± 0.069 | 0.892 ± 0.012 | 0.675 ± 0.056 |
| 3 | 4.5 | 0.949 ± 0.013 | 0.774 ± 0.062 | 0.917 ± 0.009 | 0.723 ± 0.054 |
| 5 | 6.5 | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.921 ± 0.008 | 0.726 ± 0.053 |
| 10 | 11.5 | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.921 ± 0.008 | 0.726 ± 0.053 |
| 20 | 21.5 | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.921 ± 0.008 | 0.726 ± 0.053 |
| 30 | 31.5 | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.921 ± 0.008 | 0.726 ± 0.053 |
| 60 | 61.5 | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.921 ± 0.008 | 0.726 ± 0.053 |
| inf | inf | 0.951 ± 0.013 | 0.776 ± 0.059 | 0.921 ± 0.008 | 0.726 ± 0.053 |

## Streaming LLR (retrained, ONNX INT8 dyn., batch 1, test_wild, pooled latencies of detected intakes)

| Step (s) | Budget | Sensitivity | Specificity | FA/h | Latency first: median / p95 (s) | Latency run end: median / p95 (s) | From G3 end, first: median / p95 (s) |
|---|---|---|---|---|---|---|---|
| 2 | fa05 | 0.961 ± 0.014 | 0.951 ± 0.016 | 0.497 ± 0.208 | -0.7 / 1.8 | 34.4 / 41.9 | 0.7 / 2.6 |
| 2 | fa10 | 0.965 ± 0.014 | 0.949 ± 0.018 | 0.952 ± 0.348 | -1.1 / 1.7 | 34.3 / 41.9 | 0.7 / 2.3 |
| 5 | fa05 | 0.962 ± 0.015 | 0.956 ± 0.015 | 0.528 ± 0.259 | -0.1 / 4.4 | 34.9 / 43.8 | 1.9 / 5.0 |
| 5 | fa10 | 0.968 ± 0.014 | 0.945 ± 0.023 | 0.983 ± 0.260 | -0.2 / 4.4 | 34.9 / 43.9 | 1.9 / 4.7 |
| 10 | fa05 | 0.959 ± 0.014 | 0.956 ± 0.016 | 0.590 ± 0.213 | 1.8 / 8.9 | 33.3 / 46.9 | 4.7 / 9.5 |
| 10 | fa10 | 0.962 ± 0.015 | 0.946 ± 0.022 | 0.994 ± 0.333 | 1.8 / 8.9 | 33.3 / 46.9 | 4.5 / 9.3 |

## ONNX precisions (per replicate: size, agreement with FP32)

| Precision | Size (KiB) | argmax agreement vs FP32 (min over replicates) | max abs diff |
|---|---|---|---|
| fp32 | 660 | reference | – |
| fp16 | 334 | 0.9998 | 0.060 |
| int8_dynamic | 179 | 0.9947 | 0.891 |
