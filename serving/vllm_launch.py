#!/usr/bin/env python3
"""`vllm` CLI with a server-side default thinking budget.

  THINK_BUDGET=2048 python serving/vllm_launch.py serve <model> --reasoning-config '{"reasoning_start_str": "<think>",
      "reasoning_end_str": "</think>"}' ...

vLLM 0.27.1 can cap the thinking of one request (`thinking_token_budget`: once that many thinking tokens are out it
forces `</think>`, so the answer still fits in max_tokens), but only when the request asks for it. The grader's
requests don't, so this sets it for every chat request that leaves it unset. The organizers allowed a server-side
thinking cap that keeps thinking on and leaves max_tokens, temperature and sampling alone (docs/RULES.md, Q&A).
The patch lives in the API server, which `vllm serve` runs in this process (one API server, the default).
"""
import os
import sys


def patch_default_budget(budget: int) -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    _to_sampling_params = ChatCompletionRequest.to_sampling_params

    def to_sampling_params(self, *args, **kwargs):
        if self.thinking_token_budget is None:
            self.thinking_token_budget = budget
        return _to_sampling_params(self, *args, **kwargs)

    ChatCompletionRequest.to_sampling_params = to_sampling_params
    print(f"vllm_launch: default thinking_token_budget = {budget}", flush=True)


# vLLM starts the engine core with `spawn`, which re-imports this file in the child: keep the launch under the guard.
if __name__ == "__main__":
    budget = int(os.environ.get("THINK_BUDGET") or 0)
    if budget > 0:
        patch_default_budget(budget)
    from vllm.entrypoints.cli.main import main

    sys.argv[0] = "vllm"
    sys.exit(main())
