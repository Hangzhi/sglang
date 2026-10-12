# WIP prefix reproducer validation

**Result: all three known TweetEval cases still return HTTP 400.**

- The new manual reproducer exited 1, as expected for the unfixed bug.
- Each response reports a different denoised prefix from the required empty thought channel.
- No requests were retried. Dataset and question hashes matched the pinned input.
- Repository pre-commit hooks passed for both new files.
- The GPU run used eager execution, seed 42, BF16, TP1, Triton, and a full 256-token canvas on one RTX PRO 6000 Blackwell 96GB GPU.
- The decoding fix and its broader acceptance checks remain pending.

See [requests and raw responses](wip-prefix-results.json) and [launch command, checkpoint revision, and commit metadata](wip-prefix-metadata.json).

The tested local commit and GitHub-published commit have the same Git tree. Their commit metadata differs because the GitHub connector publishes the commit.

[Earlier upstream comparison and raw token diagnosis](../tweeteval-diagnosis/README.md).
