"""Proposers: the part of the loop that decides what to try next.

Three implementations behind one interface.

`HeuristicProposer` encodes what the measurements taught us -- narrow the
cheapest stages first, keep the bit-exact plan for narrow I/O dtypes, prefer
graph capture when the profile says launch-bound, and only consider tile
configurations that fit this GPU's shared memory. It needs no API key, and it is
what makes the loop reproducible for anyone cloning the repository.

`OpenAICompatProposer` sends the same evidence to an LLM over any
OpenAI-compatible gateway (we use NUS SoC LaaS) and parses a JSON plan back.
`AnthropicProposer` does the same over the Anthropic Messages API.

All three emit `Plan` objects, and every plan they emit goes through the same
numeric gate before it is allowed near a benchmark. That separation is the whole
safety argument: **a proposer is allowed to be wrong, the gate is not.** In
practice the LLM does propose configurations our measurements already rule out
-- bfloat16 compute, an fp16 residual stream -- and the gate rejects them without
them ever posting a time. That is the system working, not failing.
"""
from __future__ import annotations

import dataclasses
import json
import random
import os
import socket
import time
from typing import Optional, Sequence

from ..optimized import Plan
from ..ops.flash import legal_blocks
from ..search import SAFE, STAGES

MODEL = os.environ.get("KERNELFORGE_MODEL", "claude-opus-5")


def _env(suffix: str):
    """First value set among the accepted aliases for `LLM_<suffix>`.

    `LLM_*` is the documented name. `OPENAI_*` is accepted because most tools
    already export it, and `SOCLAAS_*` because it is what our own runs used.
    """
    for prefix in ("LLM", "OPENAI", "SOCLAAS"):
        val = os.environ.get(f"{prefix}_{suffix}")
        if val:
            return val
    return None


@dataclasses.dataclass
class Attempt:
    """One entry in the kernel genealogy."""
    iteration: int
    plan: Plan
    utilization: float = float("nan")
    passed: bool = False
    median_ms: float = float("nan")
    speedup: float = float("nan")
    status: str = "proposed"   # proposed | compile_error | numeric_fail | ok
    detail: str = ""
    proposer: str = ""

    def line(self) -> str:
        return (f"#{self.iteration:<3} {self.status:<13} env="
                f"{self.utilization:6.3f} {self.median_ms:9.4f}ms "
                f"{self.speedup:6.3f}x  {self.plan.name}")

    def as_history(self) -> str:
        return (f"- {self.plan.name}: {self.plan.describe()} -> {self.status}, "
                f"envelope={self.utilization:.3f}, "
                f"latency={self.median_ms:.4f}ms, speedup={self.speedup:.3f}x"
                + (f" [{self.detail[:120]}]" if self.detail else ""))


class Proposer:
    name = "base"

    def propose(self, case, spec, prof, history: Sequence[Attempt],
                stage_cost, margin: float) -> Optional[Plan]:
        raise NotImplementedError


class HeuristicProposer(Proposer):
    """Deterministic search driven by the measured evidence."""

    name = "heuristic"

    def propose(self, case, spec, prof, history, stage_cost, margin):
        tried = {a.plan.name for a in history}
        smem = spec.shared_mem_per_block_kb
        dh = case.d_model // case.num_heads
        narrow_io = case.dtype in ("float16", "bfloat16")

        base = Plan(name="wide", compute_dtype="auto", residual_dtype="float32",
                    fuse_qkv=True, attention="flash", fused_norm=True,
                    smem_kb=smem)

        # For fp16/bf16 inputs no *reassociating* plan can clear the gate, so the
        # useful directions are bit-exact structure plus graph capture -- and
        # torch.compile applied to that structure, which measured bit-exact
        # while torch.compile on the baseline does not pass at these dtypes.
        if narrow_io:
            for cand in (dataclasses.replace(SAFE, smem_kb=smem, cuda_graph=True,
                                             name="safe(exact)+graph"),
                         dataclasses.replace(SAFE, smem_kb=smem,
                                             torch_compile="max-autotune",
                                             name="exact+compile[ma]"),
                         dataclasses.replace(SAFE, smem_kb=smem,
                                             torch_compile="reduce-overhead",
                                             name="exact+compile[ro]"),
                         dataclasses.replace(SAFE, smem_kb=smem,
                                             name="safe(exact)")):
                if cand.name not in tried:
                    return cand
            return None

        # 1. establish both structural bases: the fused Triton norm is faster but
        #    is the largest single error source, so the torch-norm variant is a
        #    real alternative whenever envelope, not launches, is the constraint.
        wide_tn = dataclasses.replace(base, fused_norm=False,
                                      name="wide(torch-norm)")
        if base.name not in tried:
            return base
        if wide_tn.name not in tried:
            return wide_tn

        # 2. narrow stages cheapest-error-first, from whichever base did better
        prior = {a.plan.name: a.utilization for a in history}
        if prior.get(wide_tn.name, 1e9) < prior.get(base.name, 1e9) - 0.05:
            base = wide_tn
        order = sorted(STAGES, key=lambda s: stage_cost.get(s, float("inf")))
        running = base
        for stage in order:
            if stage_cost.get(stage, float("inf")) == float("inf"):
                continue
            running = running.with_override(stage, "float16")
            named = dataclasses.replace(
                running,
                name="fp16[" + ",".join(s for s, _ in running.overrides) + "]"
                     + ("" if running.fused_norm else "/tn"))
            if named.name not in tried:
                return named
        # 3. graph-capture the best passing plan so far
        best = max((a for a in history if a.passed),
                   key=lambda a: a.speedup if a.speedup == a.speedup else -1,
                   default=None)
        if best is not None and not best.plan.cuda_graph:
            cand = dataclasses.replace(best.plan, cuda_graph=True,
                                       name=best.plan.name + "+graph")
            if cand.name not in tried:
                return cand
        # 4. sweep the legal tile configurations of the best plan
        if best is not None:
            for blk in legal_blocks(case.seq_len, dh, smem):
                cand = dataclasses.replace(
                    best.plan, flash_block=blk,
                    name=f"{best.plan.name}|blk{blk[0]}x{blk[1]}w{blk[2]}s{blk[3]}")
                if cand.name not in tried:
                    return cand
        return None


SYSTEM_PROMPT = """You optimize GPU kernels for a Transformer inference stack.

You will be given: a hardware spec sheet, a measured bottleneck profile, the
correctness rule, and the full history of configurations already tried with
their measured results.

Reply with ONE JSON object and nothing else:

{"reasoning": "<= 3 sentences",
 "plan": {"name": "...", "compute_dtype": "auto|float16|bfloat16",
          "residual_dtype": "auto|float16|float32",
          "fuse_qkv": true, "attention": "flash|sdpa|exact",
          "fused_norm": true, "cuda_graph": false,
          "match_score_rounding": false,
          "overrides": [["ffn1","float32"]],
          "flash_block": [128, 64, 8, 3]}}

Rules you must respect:
- Never repeat a configuration already in the history.
- `flash_block` is [BLOCK_M, BLOCK_N, num_warps, num_stages] and MUST fit the
  shared memory budget in the spec sheet; you will be told which are legal.
- Correctness is checked as: pass if abs_err <= atol OR abs_err <= rtol*|ref|.
  "envelope" in the history is max(abs_err / max(atol, rtol*|ref|)); it must
  come in under the stated margin, not merely under 1.0.
- Optimize for lowest latency subject to passing the gate."""


class OpenAICompatProposer(Proposer):
    """LLM proposer over any OpenAI-compatible chat-completions gateway.

    Used here with NUS SoC LLM-as-a-Service, but the
    same class works against any endpoint exposing `POST /chat/completions`.
    Implemented on `urllib` rather than the `openai` SDK so the repository stays
    dependency-free and a reviewer can reproduce the loop without installing a
    client library.

    The model sees the hardware spec sheet, the measured bottleneck profile, the
    measured per-stage error costs, the tile configurations that actually fit
    this GPU, and every previous attempt with its measured envelope and latency.
    It is doing informed search, not blind generation -- and whatever it returns
    still has to clear the same numeric gate as everything else.
    """

    name = "llm"

    def __init__(self, base_url=None, api_key=None, model=None,
                 timeout: float = 120.0):
        from .. import secrets as _secrets
        _secrets.load()
        # Provider-neutral names first. `SOCLAAS_*` is what our own gateway
        # used and stays supported, but nothing here is specific to it -- any
        # OpenAI-compatible /v1/chat/completions endpoint works, and naming the
        # primary variable after one vendor made a general feature look tied to
        # it.
        self.base_url = (base_url or _env("BASE_URL") or "").rstrip("/")
        self.api_key = api_key or _env("API_KEY")
        self.model = model or _env("MODEL") or "default"
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "set LLM_BASE_URL and LLM_API_KEY (see .env.example)")
        self.transcript = []
        self.failures = 0
        # What the gateway actually served. Our gateway aliases several model
        # ids -- asking for `qwen3.6:27b` gets you `qwen3.8:27b`, and `default`
        # gets you `qwen3.6:35b` -- so a per-model comparison that trusts the
        # requested name attributes results to the wrong model. Every response
        # echoes a `model` field; we keep it and report it.
        self.served_models = set()

    def _post(self, payload):
        import urllib.error
        import urllib.request
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"})
        # The gateway calls out 429 and 5xx as retryable. Back off hard on those.
        #
        # This used to give up after 4 tries and ~14 seconds of waiting, which is
        # nothing against a sustained rate limit. In a nine-model bake-off that
        # cost us three entire arms: two models returned HTTP 429 on all twenty
        # attempts and were recorded as having produced no kernels, which reads
        # as "the model failed" when it means "we were throttled". A comparison
        # that silently turns a rate limit into a model's score is worse than no
        # comparison, so the ceiling here is minutes, not seconds.
        attempts = 8
        delay = 5.0
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or last:
                    raise
                # Honour Retry-After when the server sends one; it knows the
                # window better than our doubling does.
                wait = delay
                hdr = e.headers.get("Retry-After") if e.headers else None
                if hdr:
                    try:
                        wait = max(wait, float(hdr))
                    except ValueError:
                        pass
                time.sleep(min(wait, 120.0) + random.uniform(0, 1.5))
                delay *= 2
            except urllib.error.URLError as e:
                # Connection refused, DNS failure, bad TLS: a misconfigured
                # endpoint, not a busy one. Retrying cannot fix it, and eight
                # backoffs would make a judge with a typo in LLM_BASE_URL
                # wait six minutes for an error we already know is permanent.
                # Timeouts are the exception -- a slow model is worth waiting for.
                if isinstance(e.reason, socket.timeout) and not last:
                    time.sleep(min(delay, 120.0) + random.uniform(0, 1.5))
                    delay *= 2
                    continue
                raise
            except socket.timeout:
                if last:
                    raise
                time.sleep(min(delay, 120.0) + random.uniform(0, 1.5))
                delay *= 2
        raise RuntimeError("unreachable")

    def _prompt(self, case, spec, prof, history, stage_cost, margin):
        dh = case.d_model // case.num_heads
        legal = legal_blocks(case.seq_len, dh, spec.shared_mem_per_block_kb)
        costs = "\n".join("  {}: {:+.3f}".format(k, v) for k, v in
                           sorted(stage_cost.items(), key=lambda kv: kv[1]))
        hist = "\n".join(a.as_history() for a in history)
        return "\n\n".join([
            "## Hardware", spec.as_prompt_block(),
            "## Measured profile", prof.as_prompt_block() if prof else "(none)",
            "## Workload",
            ("batch={} seq_len={} d_model={} heads={} head_dim={} ffn={} "
             "layers={} causal={} padding_ratio={} io_dtype={}").format(
                case.batch_size, case.seq_len, case.d_model, case.num_heads,
                dh, case.ffn_dim, case.num_layers, case.causal,
                case.padding_ratio, case.dtype),
            "## Measured per-stage error cost "
            "(envelope added by putting that stage in fp16)",
            costs or "(not measured)",
            "## Legal flash_block configurations on this GPU\n{}".format(legal),
            "## Correctness margin\nenvelope must be <= {}".format(margin),
            "## History", hist or "(nothing tried yet)",
            "Propose the next configuration.",
        ])

    def _note_served(self, data) -> None:
        served = data.get("model")
        if served:
            self.served_models.add(served)

    def raw_completion(self, system: str, prompt: str, max_tokens: int = 2400,
                       temperature: float = 0.5) -> str:
        """Free-form completion, used by the kernel code-generation loop."""
        data = self._post({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        })
        text = data["choices"][0]["message"]["content"] or ""
        self._note_served(data)
        self.transcript.append({"kind": "codegen", "prompt": prompt,
                                "response": text,
                                "usage": data.get("usage", {}),
                                "model": self.model})
        return text

    def propose(self, case, spec, prof, history, stage_cost, margin):
        prompt = self._prompt(case, spec, prof, history, stage_cost, margin)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 900,
            "reasoning_effort": "none",
        }
        try:
            data = self._post(payload)
            text = data["choices"][0]["message"]["content"] or ""
            self._note_served(data)
            usage = data.get("usage", {})
        except Exception as e:
            self.failures += 1
            self.transcript.append(
                {"case": case.name, "prompt": prompt,
                 "error": "{}: {}".format(type(e).__name__, e)})
            return None

        self.transcript.append({"case": case.name, "prompt": prompt,
                                "response": text, "usage": usage,
                                "model": self.model})
        plan = parse_plan(text, spec.shared_mem_per_block_kb)
        if plan is None:
            self.failures += 1
            return None
        # Never let the model silently repeat itself: a duplicate name would
        # collide in the dispatch table and waste an iteration re-measuring a
        # point we already know.
        if plan.name in {a.plan.name for a in history}:
            plan = dataclasses.replace(
                plan, name="{}#{}".format(plan.name, len(history) + 1))
        return plan


class AnthropicProposer(Proposer):
    """LLM proposer over the Anthropic Messages API. Requires ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model: str = MODEL):
        import anthropic  # raises if the SDK is absent
        self.client = anthropic.Anthropic()
        self.model = model
        self.transcript = []
        self.failures = 0
        # Kept for interface parity with the gateway client, which records what
        # was actually served because that gateway aliases model ids.
        self.served_models = {model}

    def propose(self, case, spec, prof, history, stage_cost, margin):
        helper = OpenAICompatProposer.__dict__["_prompt"]
        prompt = helper(self, case, spec, prof, history, stage_cost, margin)
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if b.type == "text")
        self.transcript.append({"case": case.name, "prompt": prompt,
                                "response": text})
        return parse_plan(text, spec.shared_mem_per_block_kb)


def parse_plan(text: str, smem_kb: float) -> Optional[Plan]:
    """Parse a model reply into a Plan, tolerating fenced code blocks."""
    body = text.strip()
    if "```" in body:
        body = body.split("```")[1]
        if body.startswith("json"):
            body = body[4:]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        blob = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None
    p = blob.get("plan", blob)
    fb = p.get("flash_block")
    fields = {f.name for f in dataclasses.fields(Plan)}
    kwargs = {k: v for k, v in p.items() if k in fields}
    kwargs["overrides"] = tuple(tuple(o) for o in p.get("overrides", []) or [])
    kwargs["flash_block"] = tuple(fb) if fb else None
    kwargs["smem_kb"] = smem_kb
    kwargs.setdefault("name", "llm")
    try:
        return Plan(**kwargs)
    except TypeError:
        return None


def build(provider: str) -> Proposer:
    """Select a proposer. `auto` prefers an LLM when one is configured."""
    from .. import secrets as _secrets
    _secrets.load()

    if provider == "heuristic":
        return HeuristicProposer()

    if provider in ("llm", "soclaas", "openai", "auto"):
        try:
            if _env("API_KEY"):
                return OpenAICompatProposer()
        except Exception as e:
            if provider != "auto":
                raise
            print("[agent] LLM proposer unavailable ({}: {}); "
                  "falling back to the heuristic proposer".format(
                      type(e).__name__, e))

    if provider in ("anthropic", "auto"):
        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                return AnthropicProposer()
        except Exception as e:
            if provider == "anthropic":
                raise
            print("[agent] Anthropic proposer unavailable ({}); "
                  "using the heuristic proposer".format(type(e).__name__))

    if provider != "auto":
        raise SystemExit("unknown or unavailable proposer: " + provider)
    return HeuristicProposer()
