# TweetEval prefix diagnosis and coverage follow-up

**The three errors come from invalid generated prefixes. All three also occur in upstream code without either PR.**

Date: **11 October 2026, America/Vancouver**. Raw server logs use 12 October UTC. Code heads and checkpoint details are in [summary.json](summary.json) and the original [validation manifest](../manifest.json).

## Why the API returns HTTP 400

- The score reader requires prefix IDs `[100, 45518, 107, 101]`.
- These decode to `<|channel>thought\n<channel|>`.
- The answer score belongs to position 4. The first position is a channel marker.
- All three failing requests start with the first three expected tokens. The fourth token is wrong.
- The score reader raises `ValueError` on a prefix mismatch. The shared HTTP handler converts that exception to HTTP 400 (`BadRequest`). The request itself is well-formed.

| Frozen sample ID (short) | Actual fourth token | Actual first five tokens, decoded |
|---|---|---|
| `6e64a2415fdf` | `1018` (`**`) | `<|channel>thought\n**F` |
| `2e1ab45c412a` | `1` (`<eos>`) | `<|channel>thought\n<eos>N` |
| `f950a583343b` | `1` (`<eos>`) | `<|channel>thought\n<eos>S` |

None of the full 256-token outputs contains the required closing channel token. This is not a longer thought block with an answer that can be found later.

## Controlled checks

Same frozen samples, native prompt IDs, checkpoint, BF16 GPU and seed 42. No retries were used to select a successful answer. Each row below is a separate, named experiment; the original benchmark scores remain unchanged.

| Experiment | `6e64…` | `2e1a…` | `f950…` |
|---|---|---|---|
| FDFO + CUDA graph, original defaults | 400 | 400 | 400 |
| Synchronous eager, original defaults | 400 | 400 | 400 |
| Eager, early stopping disabled, 48 steps | 400 | 200 | 400 |
| Eager, early stopping disabled, 96 steps | 400 | 400 | 400 |

- Returning 5, 64 or 256 tokens preserves the same bad prefix in the original configuration.
- Turning score collection off preserves the generated tokens in every experiment above.
- An isolated, unchanged upstream checkout at `20c49f12c9a1261ea0ba9b5a2b206613661494f0` produces the same first eight tokens for all three requests. Neither PR is present in that checkout.
- The model's denoiser scores support the bad token; this is not a token transport or decoding-display mismatch. The raw selected/candidate scores are archived.

**Result:** no reliable fix was found by changing output length, execution mode or the tested stopping/step settings. The failure predates this stack. These checks do not isolate whether checkpoint behavior or another part of the upstream denoising implementation is the ultimate cause. A reference-decoder comparison would be needed for that attribution.

The prefix guard remains unchanged. Accepting an unclosed thought channel or early EOS would weaken the score contract. A full fix needs validated prefix handling in decoding or a separately specified scoring layout, followed by a full-data run. Changing only the HTTP status would not fix the missing valid answer prefix.

## Why the tests say Gemma4

- The implementation class is `Gemma4Renoise`. Its supported architecture is `DiffusionGemmaForBlockDiffusion`.
- The `test_gemma4_*` files test this implementation. They are DiffusionGemma tests.
- The PR1 screenshot shows changed files only. Shared tests can run without appearing in that diff.
- PR2 has decision API tests. The earlier full-stack run also includes shared scoring and KV-cache tests.

The follow-up ran **85 CPU tests and 14 subtests**, all passed:

| Existing test file | Tests passed |
|---|---:|
| `dllm/algorithm/test_joint_threshold_indel.py` | 29 |
| `mem_cache/test_dllm_fdfo_kv_reuse.py` | 6 |
| `managers/test_setwise_score_mixin.py` | 50 |

These overlap the earlier 129-test suite by 56 tests. They are not 85 additional unique tests.

A separate CPU smoke script passed **six configurations**: `LowConfidence`, `JointThreshold` and `JointThresholdInDel`, each in synchronous and FDFO modes. It checks completed tokens, preserved prompt tokens, the final KV-commit forward and the absence of Gemma-specific output fields. It exercises the shared run/finalize path. It is a validation script, not a newly committed CI test. These CPU checks do not qualify other model weights end to end on a GPU.

## Evidence and replay

[Download the diagnosis archive](diagnosis.tar.gz?raw=true). It contains exact request/prompt IDs, raw model outputs, configuration files, scripts, server logs and the new test transcript. Check [SHA256SUMS](SHA256SUMS) after download.

The original model checkpoint and environment are unchanged. The archive preserves tested absolute paths under `/home/ubuntu`; adapt paths for another workspace. `debug_tweeteval.py` reconstructs prompts through the committed SystemOne renderer. `check_prefix_variant.py` compares plain generation, scored generation and SystemOne. `launch_debug_variant.py` records each launch configuration. `check_baseline.py` replays the same prompt IDs against the upstream-only server.

```bash
# With the pinned stack server running, as in the original validation README:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/home/ubuntu/sglang/python \
  /home/ubuntu/diffusiongemma-e2e/venv/bin/python \
  /home/ubuntu/diffusiongemma-e2e/debug_tweeteval.py

# Shared regression run from the SGLang checkout:
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=python \
  /home/ubuntu/diffusiongemma-e2e/venv/bin/python -m pytest -q --disable-warnings \
  test/registered/unit/dllm/algorithm/test_joint_threshold_indel.py \
  test/registered/unit/mem_cache/test_dllm_fdfo_kv_reuse.py \
  test/registered/unit/managers/test_setwise_score_mixin.py
```

## LOC accounting

[loc.json](loc.json) is generated with `git diff --numstat`. PR1 is measured against its base; PR2 is measured against PR1. Markdown files count as documentation, including the README under `python/sglang/srt`. Counts include comments and blank lines. Neither feature branch changed during this diagnosis.
