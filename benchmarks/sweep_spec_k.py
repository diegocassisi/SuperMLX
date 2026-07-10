"""Sweep K values for speculative decoding to find optimal speedup."""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

TARGET = "mlx-community/Agents-A1-3bit"
DRAFT = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
PROMPT = "Write a Python function that implements binary search on a sorted list. Include docstring and edge case handling."
ROUNDS = 50


def forward(model, input_ids, cache):
    text_model = getattr(model, "language_model", model)
    inner = text_model.model
    h = inner.embed_tokens(input_ids)
    fa_mask = create_attention_mask(h, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
    for layer, lc in zip(inner.layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=lc)
    h = inner.norm(h)
    if text_model.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(h)
    return text_model.lm_head(h)


def save_cache(cache):
    saved = []
    for c in cache:
        if isinstance(c, KVCache):
            saved.append(("kv", c.offset, c.keys, c.values))
        else:
            state_copy = [mx.array(s) if s is not None else None for s in c.state]
            saved.append(("arrays", state_copy))
    return saved


def restore_cache(cache, saved):
    for c, entry in zip(cache, saved):
        if entry[0] == "kv":
            _, offset, keys, values = entry
            c.offset = offset
            c.keys = keys
            c.values = values
        else:
            _, state_copy = entry
            c.state = state_copy


def run_baseline(target, tokenizer, prompt_tokens, max_tokens):
    cache = make_prompt_cache(target)
    logits = forward(target, mx.array([prompt_tokens]), cache)
    mx.eval(logits)
    generated = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        logits = forward(target, mx.array([[token]]), cache)
        mx.eval(logits)
    return len(generated), time.perf_counter() - t0


def run_speculative(target, draft, tokenizer, prompt_tokens, k, max_rounds):
    target_cache = make_prompt_cache(target)
    draft_cache = make_prompt_cache(draft)
    input_ids = mx.array([prompt_tokens])
    target_logits = forward(target, input_ids, target_cache)
    draft_logits = forward(draft, input_ids, draft_cache)
    mx.eval(target_logits, draft_logits)

    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    pos0_match = 0
    pos0_total = 0

    t0 = time.perf_counter()

    for _ in range(max_rounds):
        target_token = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())
        total_tokens += 1
        if target_token == tokenizer.eos_token_id:
            break

        saved_target = save_cache(target_cache)
        saved_draft = save_cache(draft_cache)

        # Draft K tokens
        draft_tokens = []
        current = target_token
        for _ in range(k):
            dl = forward(draft, mx.array([[current]]), draft_cache)
            mx.eval(dl)
            d = int(mx.argmax(dl[:, -1, :], axis=-1).item())
            draft_tokens.append(d)
            current = d

        total_drafted += k

        # Verify
        verify_seq = [target_token] + draft_tokens[:-1]
        verify_logits = forward(target, mx.array([verify_seq]), target_cache)
        mx.eval(verify_logits)

        accepted = 0
        correction = None
        for i in range(k):
            t_next = int(mx.argmax(verify_logits[:, i, :], axis=-1).item())
            if i == 0:
                pos0_total += 1
                if t_next == draft_tokens[0]:
                    pos0_match += 1
            if t_next == draft_tokens[i]:
                accepted += 1
            else:
                correction = t_next
                break

        total_accepted += accepted
        total_tokens += accepted + (1 if correction is not None else 0)

        # Restore and replay
        restore_cache(target_cache, saved_target)
        restore_cache(draft_cache, saved_draft)
        replay = [target_token] + draft_tokens[:accepted]
        if correction is not None:
            replay.append(correction)
        target_logits = forward(target, mx.array([replay]), target_cache)
        draft_logits = forward(draft, mx.array([replay]), draft_cache)
        mx.eval(target_logits, draft_logits)

    elapsed = time.perf_counter() - t0
    return total_tokens, elapsed, total_accepted, total_drafted, pos0_match, pos0_total


def main():
    print(f"Loading models...")
    target, tokenizer = load(TARGET)
    mx.eval(target.parameters())
    draft, _ = load(DRAFT)
    mx.eval(draft.parameters())
    print(f"GPU: {mx.get_active_memory()/1e9:.2f} GB\n")

    messages = [{"role": "user", "content": PROMPT}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True
    )

    # Baseline
    max_tok = ROUNDS * 5  # enough tokens for comparison
    bl_tokens, bl_time = run_baseline(target, tokenizer, prompt_tokens, max_tok)
    bl_tps = bl_tokens / bl_time
    print(f"BASELINE: {bl_tokens} tokens in {bl_time:.2f}s = {bl_tps:.1f} tok/s\n")

    # Sweep K values
    print(f"{'K':>3} | {'Tokens':>6} | {'Time':>6} | {'tok/s':>6} | {'Speedup':>7} | {'Accept':>7} | {'Pos0':>6} | {'tok/round':>9}")
    print("-" * 75)

    for k in [1, 2, 3, 4, 6, 8]:
        tokens, elapsed, accepted, drafted, p0_match, p0_total = run_speculative(
            target, draft, tokenizer, prompt_tokens, k, ROUNDS
        )
        tps = tokens / elapsed
        speedup = tps / bl_tps
        acc = accepted / drafted * 100 if drafted > 0 else 0
        p0 = p0_match / p0_total * 100 if p0_total > 0 else 0
        tpr = tokens / ROUNDS

        print(f"{k:>3} | {tokens:>6} | {elapsed:>5.2f}s | {tps:>5.1f} | {speedup:>6.2f}x | {acc:>5.1f}% | {p0:>4.0f}% | {tpr:>8.1f}")


if __name__ == "__main__":
    main()
