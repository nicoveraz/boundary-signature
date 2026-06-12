"""Phase C batched-decode validation against mlx built from main.

Decisive test for the mx.fast.rope batch>=2,seq_len==1 fix (PR #3498):
batched decode of N rows sharing a prefix must, at temperature=0,
produce identical token streams across all rows AND match the serial
single-sample argmax path. The pre-fix kernel made rows 1+ diverge.

Usage: uv run python /tmp/validate_phase_c.py
"""
import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
PREFIX_TEXT = (
    "You are a careful clinical reasoner. A 58-year-old presents with "
    "chest pain radiating to the left arm, diaphoresis, and ECG showing "
    "ST elevation. Step by step, the most likely diagnosis is"
)
N = 3
MAX_TOKENS = 24


def rope_repro() -> bool:
    mx.random.seed(0)
    q1 = mx.random.normal((1, 8, 1, 64))
    q = mx.concatenate([q1, q1], axis=0)
    out = mx.fast.rope(q, 64, traditional=False, base=10000.0, scale=1.0, offset=5)
    mx.eval(out)
    diff = mx.max(mx.abs(out[0] - out[1])).item()
    print(f"[rope repro] mlx={mx.__version__} max-abs-diff={diff}")
    return diff == 0.0


def serial_argmax(model, prefix_ids, max_tokens):
    cache = make_prompt_cache(model)
    logits = model(mx.array(prefix_ids)[None, :], cache=cache)
    last = logits[:, -1, :]
    out = []
    for _ in range(max_tokens):
        tok = int(mx.argmax(last[0]).item())
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)
        last = logits[:, -1, :]
    return out


def batched_argmax(model, prefix_ids, n, max_tokens):
    cache = make_prompt_cache(model)
    prefix = mx.array(prefix_ids)[None, :]
    prefix = mx.repeat(prefix, n, axis=0)  # [N, L]
    logits = model(prefix, cache=cache)
    last = logits[:, -1, :]  # [N, vocab]
    rows = [[] for _ in range(n)]
    for _ in range(max_tokens):
        toks = mx.argmax(last, axis=-1)  # [N]
        mx.eval(toks)
        tlist = [int(t) for t in toks.tolist()]
        for i, t in enumerate(tlist):
            rows[i].append(t)
        nxt = mx.array(tlist)[:, None]  # [N, 1]
        logits = model(nxt, cache=cache)
        last = logits[:, -1, :]
    return rows


def main():
    fixed = rope_repro()
    if not fixed:
        print("FAIL: rope kernel still buggy — this mlx build does not carry the fix.")
        sys.exit(1)

    print(f"[load] {MODEL}")
    model, tokenizer = load(MODEL)
    prefix_ids = tokenizer.encode(PREFIX_TEXT)

    print("[serial] argmax decode")
    ser = serial_argmax(model, prefix_ids, MAX_TOKENS)
    print("[batched] argmax decode, N=", N)
    bat = batched_argmax(model, prefix_ids, N, MAX_TOKENS)

    all_rows_equal = all(r == bat[0] for r in bat)
    matches_serial = bat[0] == ser
    print(f"\nserial:    {ser}")
    for i, r in enumerate(bat):
        print(f"batched[{i}]: {r}")
    print(f"\nall batched rows identical: {all_rows_equal}")
    print(f"batched row0 == serial:     {matches_serial}")
    if all_rows_equal and matches_serial:
        print("\nPASS: Phase C batched decode is correct against this mlx build.")
        sys.exit(0)
    print("\nFAIL: batched decode diverges — Phase C still blocked.")
    sys.exit(1)


if __name__ == "__main__":
    main()
