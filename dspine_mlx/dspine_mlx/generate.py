from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx

from .bridge import build_serial_tables
from .model import load_draft, quantize_draft
from .propose import RefinementTail, filter_target_probs
from .sampler import greedy_verify, sample_from_probs, sampled_verify
from .target import TargetAdapter


@dataclass
class GenStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    rounds: int = 0
    accepted_drafts: int = 0
    committed_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    draft_seconds: float = 0.0
    tail_seconds: float = 0.0
    verify_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    block_size: int = 0

    @property
    def decode_tps(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0

    @property
    def accepted_per_round(self) -> float:
        return self.accepted_drafts / self.rounds if self.rounds else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.committed_tokens / self.rounds if self.rounds else 0.0


def _sample_target(logits: mx.array, inv_temp: float, top_k: int, top_p: float, min_p: float) -> int:
    probs = mx.softmax(logits.astype(mx.float32) * float(inv_temp), axis=-1)
    probs = filter_target_probs(probs[None], top_k, top_p, min_p)[0]
    u = mx.random.uniform()
    return int(sample_from_probs(probs, u).item())


class DSpineGenerator:
    def __init__(self, target: TargetAdapter, draft, *, block_size: int | None = None, profile: bool = False):
        if target.vocab_size != draft.vocab_size:
            raise ValueError(
                f"target vocab {target.vocab_size} != draft vocab {draft.vocab_size}"
            )
        if target.hidden_size != draft.hidden_size:
            raise ValueError(
                f"target hidden {target.hidden_size} != draft hidden {draft.hidden_size}"
            )
        self.target = target
        self.draft = draft
        self.profile = bool(profile)
        if block_size is not None:
            draft.set_block_size(block_size)
        self.block_size = int(draft.block_size)
        bridge = draft.selected_prefix_bridge
        t0 = time.perf_counter()
        self.tg, self.tsv = build_serial_tables(
            head_rows=target.head.rows_range,
            vocab_size=target.vocab_size,
            norm_w=draft.norm.weight,
            write_w=bridge.feature_write_weights[-1],
            pv_w=bridge.packed_weight[bridge.rank :],
            codebook=bridge.codebook,
        )
        self.tables_seconds = time.perf_counter() - t0
        self.tail = RefinementTail(draft, target.head, self.tg, self.tsv)

    def set_block_size(self, block_size: int) -> None:
        self.draft.set_block_size(block_size)
        self.block_size = int(block_size)

    def generate(
        self,
        prompt_tokens,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        stop_ids=(),
        on_tokens=None,
    ) -> tuple[list[int], GenStats]:
        prompt = (
            prompt_tokens
            if isinstance(prompt_tokens, mx.array)
            else mx.array([list(prompt_tokens)], dtype=mx.int32)
        )
        if prompt.ndim == 1:
            prompt = prompt[None]
        stop = set(int(x) for x in stop_ids)
        greedy = temperature <= 0
        inv_temp = 1.0 / max(float(temperature), 1e-5)
        K = int(self.draft.head_topk)
        chain_top_k = K if top_k <= 0 else min(max(int(top_k), 1), K)
        target_top_k = self.target.vocab_size if top_k <= 0 else int(top_k)

        stats = GenStats(prompt_tokens=int(prompt.shape[1]), block_size=self.block_size)
        t0 = time.perf_counter()
        self.target.reset()
        logits, hidden = self.target.prefill(prompt)
        mx.eval(logits, hidden)
        stats.prefill_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        if greedy:
            anchor = int(mx.argmax(logits[0, -1]).item())
        else:
            anchor = _sample_target(
                logits[0, -1], inv_temp, target_top_k, top_p, min_p
            )
        tokens = [anchor]
        done = anchor in stop or len(tokens) >= max_tokens

        draft_caches = self.draft.make_cache()
        while not done:
            remaining = max_tokens - len(tokens)
            round_T = min(self.block_size, remaining + 1)
            if round_T < 2:
                break

            block = mx.array(
                [[anchor] + [self.draft.mask_token_id] * (round_T - 1)],
                dtype=mx.int32,
            )
            t_phase = time.perf_counter()
            out, stash = self.draft(hidden, block, draft_caches)
            if self.profile:
                mx.eval(out)
                stats.draft_seconds += time.perf_counter() - t_phase

            rows = out[0, 1:]
            t_phase = time.perf_counter()
            if greedy:
                drafts, cand = self.tail.greedy(rows, stash, anchor)
                q_rows = None
            else:
                uniforms = mx.random.uniform(shape=(round_T - 1,))
                drafts, q_rows, cand = self.tail.sampled(
                    rows,
                    stash,
                    anchor,
                    inv_temp=inv_temp,
                    top_k=chain_top_k,
                    top_p=top_p,
                    min_p=min_p,
                    uniforms=uniforms,
                )
            if self.profile:
                mx.eval(drafts, cand)
                stats.tail_seconds += time.perf_counter() - t_phase
            else:
                # Let the draft/tail run while the verify graph is built.
                if greedy:
                    mx.async_eval(drafts)
                else:
                    mx.async_eval(drafts, q_rows, cand)

            verify_tokens = mx.concatenate(
                [mx.array([[anchor]], dtype=mx.int32), drafts[None]], axis=1
            )
            t_phase = time.perf_counter()
            vlogits, vhidden = self.target.verify(verify_tokens)
            if greedy:
                accept, bonus = greedy_verify(vlogits[0], drafts)
            else:
                accept, bonus = sampled_verify(
                    vlogits[0],
                    drafts,
                    cand,
                    q_rows,
                    inv_temp=inv_temp,
                    top_k=target_top_k,
                    top_p=top_p,
                    min_p=min_p,
                    u_accept=mx.random.uniform(shape=(round_T - 1,)),
                    u_final=mx.random.uniform(),
                    u_res=mx.random.uniform(),
                )
            mx.eval(accept, bonus, drafts, vhidden)
            if self.profile:
                stats.verify_seconds += time.perf_counter() - t_phase

            accept_i = int(accept.item())
            bonus_i = int(bonus.item())
            draft_list = drafts.tolist()
            new_tokens = draft_list[:accept_i] + [bonus_i]
            new_tokens = new_tokens[: max_tokens - len(tokens)]

            eos_at = next(
                (i for i, t in enumerate(new_tokens) if t in stop), None
            )
            if eos_at is not None:
                new_tokens = new_tokens[: eos_at + 1]
                done = True

            if not new_tokens:
                done = True
                break

            tokens.extend(new_tokens)
            stats.rounds += 1
            stats.accepted_drafts += min(accept_i, len(new_tokens))
            stats.committed_tokens += len(new_tokens)
            if on_tokens is not None:
                on_tokens(new_tokens)
            if len(tokens) >= 256 and len(tokens) % 256 < len(new_tokens):
                mx.clear_cache()
            if len(tokens) >= max_tokens:
                done = True
                break

            self.target.trim(round_T - accept_i - 1)
            hidden = vhidden[:, : accept_i + 1, :]
            anchor = new_tokens[-1]

        stats.generated_tokens = len(tokens)
        stats.decode_seconds = time.perf_counter() - t0
        stats.peak_memory_gb = mx.get_peak_memory() / 1e9
        return tokens, stats


def baseline_generate(
    target: TargetAdapter,
    prompt_tokens,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    stop_ids=(),
) -> tuple[list[int], GenStats]:
    """Token-at-a-time autoregressive baseline on the same target."""
    prompt = (
        prompt_tokens
        if isinstance(prompt_tokens, mx.array)
        else mx.array([list(prompt_tokens)], dtype=mx.int32)
    )
    if prompt.ndim == 1:
        prompt = prompt[None]
    stop = set(int(x) for x in stop_ids)
    greedy = temperature <= 0
    inv_temp = 1.0 / max(float(temperature), 1e-5)
    target_top_k = target.vocab_size if top_k <= 0 else int(top_k)

    stats = GenStats(prompt_tokens=int(prompt.shape[1]), block_size=1)
    t0 = time.perf_counter()
    target.reset()
    logits, _ = target.prefill(prompt, capture=False)
    mx.eval(logits)
    stats.prefill_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    if greedy:
        token = int(mx.argmax(logits[0, -1]).item())
    else:
        token = _sample_target(logits[0, -1], inv_temp, target_top_k, top_p, min_p)
    tokens = [token]
    while len(tokens) < max_tokens and token not in stop:
        logits = target.step(token)
        mx.eval(logits)
        if greedy:
            token = int(mx.argmax(logits[0, -1]).item())
        else:
            token = _sample_target(
                logits[0, -1], inv_temp, target_top_k, top_p, min_p
            )
        tokens.append(token)
    stats.generated_tokens = len(tokens)
    stats.decode_seconds = time.perf_counter() - t0
    stats.peak_memory_gb = mx.get_peak_memory() / 1e9
    return tokens, stats


def _load_tokenizer_eos(tokenizer) -> set[int]:
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos:
        return set(int(x) for x in eos)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return {int(eos_id)} if eos_id is not None else set()


def main() -> None:
    p = argparse.ArgumentParser("dspine-generate")
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--prompt", default="What is 1+1? Answer briefly.")
    p.add_argument("--raw", action="store_true", help="do not apply the chat template")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quantize-draft", action="store_true")
    p.add_argument("--draft-bits", type=int, default=8)
    p.add_argument("--draft-group-size", type=int, default=64)
    p.add_argument("--draft-q-mode", default="affine")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()

    mx.random.seed(args.seed)
    draft, cfg = load_draft(args.draft)
    if args.quantize_draft:
        draft = quantize_draft(
            draft,
            bits=args.draft_bits,
            group_size=args.draft_group_size,
            mode=args.draft_q_mode,
        )
    target = TargetAdapter.load(args.target, cfg.target_layer_ids)
    draft.bind(target.model)
    tokenizer = target.tokenizer

    if args.raw:
        prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    stop_ids = _load_tokenizer_eos(tokenizer)
    gen = DSpineGenerator(
        target, draft, block_size=args.block_size, profile=args.profile
    )
    print(f"[tables] {gen.tables_seconds:.2f}s")
    tokens, stats = gen.generate(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        stop_ids=stop_ids,
    )
    print(tokenizer.decode(tokens))
    print(
        f"[dspine] block={stats.block_size} tokens={stats.generated_tokens} "
        f"rounds={stats.rounds} accept/round={stats.accepted_per_round:.2f} "
        f"tokens/round={stats.tokens_per_round:.2f} "
        f"decode={stats.decode_tps:.2f} tok/s "
        f"prefill={stats.prefill_seconds:.2f}s "
        f"peak={stats.peak_memory_gb:.1f}GB"
    )
    if args.profile:
        print(
            f"[profile] draft={stats.draft_seconds:.2f}s "
            f"tail={stats.tail_seconds:.2f}s verify={stats.verify_seconds:.2f}s"
        )
    if args.baseline:
        btokens, bstats = baseline_generate(
            target,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            stop_ids=stop_ids,
        )
        print(
            f"[baseline] tokens={bstats.generated_tokens} "
            f"decode={bstats.decode_tps:.2f} tok/s "
            f"speedup={stats.decode_tps / max(bstats.decode_tps, 1e-9):.2f}x"
        )


if __name__ == "__main__":
    main()
