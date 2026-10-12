# DiffusionGemma validation — 12 October 2026

Evidence for [PR1 #41232](https://github.com/sgl-project/sglang/pull/41232) and [PR2 #43752](https://github.com/sgl-project/sglang/pull/43752). This branch holds validation artifacts, not a third PR.

## Results

| Dataset | Correct / all | Valid responses | Accuracy |
|---|---:|---:|---:|
| JevBench | 199 / 231 | 231 | 86.15% |
| Emojify | 41 / 50 | 50 | 82.00% |
| TweetEval | 300 / 1000 | 997 | 30.00% |
| Flowers (image input) | 98 / 100 | 100 | 98.00% |

Three TweetEval requests returned HTTP 400 because the denoised canvas did not start with the required empty-thought prefix. They are retained as incorrect in the denominator. See [failures.json](failures.json). No retries or cherry-picked removals. These are model-format failures, not passing decisions. Candidate probabilities are relative scores, not calibrated correctness probabilities.

[checks.txt](checks.txt) lists API/unit checks. [manifest.json](manifest.json) records model revision, hardware, settings, tested and published trees. [summary.json](summary.json) includes observed serial latency (not a comparative performance claim). [Download all evidence](evidence.tar.gz?raw=true), then extract with `tar -xzf evidence.tar.gz`. The archive contains per-example predictions, raw HTTP JSON, complete test transcripts, server logs, launch commands, environment versions and the exact validation scripts. [SHA256SUMS](SHA256SUMS) verifies the files.

## What was checked

- Standalone PR1: 48 tests / 45 subtests; six real raw-score requests across all datasets; text and image; mixed candidate IDs in a real batch; streaming preserves all rows; requesting scores preserves generated tokens; input/prompt logprobs correctly rejected.
- Full stack: 129 tests / 152 subtests. Includes a regression that first failed when retraction retained unfinished canvas state, then passed after fixing reset while preserving emitted tokens and scores.
- Committed decision replay methods passed in eager and FDFO CUDA graph modes: `/v1/decisions` choice/yes_no/score, `/v1/systemone`, `/v1/score`, raw `/generate`. Replayed probabilities and raw score softmax agree within 2e-4 at temperature 1.7; label mass and zero decision output-token usage checked.
- CUDA graph decode also exercised with graph batch sizes 1, 2, 4 and mixed candidate lists. See server logs for actual graph use. Context prefill graphs were disabled.
- Repository-pinned pre-commit hooks passed independently for PR1 and full stack.

## Reproduce

The archived scripts preserve the exact tested paths under `/home/ubuntu`; adapt them if using another workspace. Use a suitable NVIDIA GPU, the archived dependency versions and checkpoint `google/diffusiongemma-26B-A4B-it` at `f7f5b7f5fa82ffc52addd066915886d497f5517b`, downloaded to `/home/ubuntu/models/diffusiongemma-26B-A4B-it`.

1. Clone SGLang from `Hangzhi/sglang` to `/home/ubuntu/sglang`. Fetch the two feature branches. Checkout published PR1 `16b1b8de683447385badc5c5989923fec31bd852` for standalone checks, or PR2 `6d92bcc6b0d198025fa2884dfc1a35647da3b6f9` for the full run. Their entire trees equal the locally tested revisions in the manifest. The rebase used Git Town; local stack parents are main → logprobs → decisions.
2. Clone `Hangzhi/diffusion-jev-sglang` at `a6dce88a2fc9c11b1df44d03033f1ebafd75c504` to `/home/ubuntu/diffusion-jev-sglang`. Use its `scripts/prepare_emoji_benchmark.py` and `scripts/prepare_flowers.py` to prepare data. Clone JevBench at `c6004e008ffba24aec091261ca1a5c02f7324702` into `/home/ubuntu/jevbench`. The harness verifies JevBench SHA256s, both emoji test-file hashes and the 100 flower IDs against the reference reports. Install the native PR code, not the reference repository's historical readout patch.
3. Extract the archive into `/home/ubuntu/diffusiongemma-e2e`; create its `venv` using `engine-requirements.txt` and compare `environment.txt`. Launch settings are in each `*-launch.json`; `launch.py` also sets required environment variables. Use the following commands (benchmark output directory must be new):

```bash
cd /home/ubuntu/diffusiongemma-e2e
venv/bin/python verify.py data
venv/bin/python launch.py repeat-graphs --graphs --max-running 1
venv/bin/python wait_ready.py repeat-graphs
PYTHONPATH=/home/ubuntu/sglang/python venv/bin/python replay.py replay-repeat.json
venv/bin/python verify.py raw --output repeat-raw
venv/bin/python verify.py benchmark --output repeat-benchmark
```

For standalone PR1, launch without `--graphs` using `--max-running 4`, run `verify.py raw`, then `batch_stream.py <raw-output-dir>`. For graph batching, restart the full stack with `--graphs --max-running 4`, run raw and batch/stream checks. Stop each server's recorded process group before launching the next.

The replay harness executes the committed test methods against an already running server and replaces only class setup/teardown. Therefore it validates the endpoint behavior but does not claim a successful CI-runner launch of the test fixture. Launch here used BF16, Triton, context 8192, TP1 on one RTX PRO 6000 Blackwell 96GB, PyTorch 2.14.1+cu130 and Transformers 5.19.0. Model uses the full 256-token canvas, default denoising schedule, sampler seed 42.

Unit command: from SGLang, run `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=python /home/ubuntu/diffusiongemma-e2e/venv/bin/python -m pytest -q --disable-warnings` followed by `test/registered/unit/dllm/test_gemma4_logprobs.py`, `test_gemma4_renoise.py`, `test_gemma4_uniform_lifecycle.py`, `test_gemma4_cuda_graph.py` (all in that same directory) and `test/registered/unit/mem_cache/test_dllm_fdfo_kv_reuse.py`. Full stack adds `test/registered/unit/entrypoints/openai/test_serving_decisions.py` and `test/registered/unit/managers/test_setwise_score_mixin.py`. File lists and transcripts are archived.

## Limits

This reuses the reference data, not identical historical prompts or hardware. The native decision endpoint generates its own prompts; accuracy/latency are fresh observations, not direct A100 performance comparisons. No forced GPU memory-pressure retraction or multi-GPU run was performed; retraction behavior has unit coverage. Unsupported prefix output is rejected rather than scored at a wrong position. Model quality on TweetEval remains low. This evidence does not establish production load performance or calibration.
