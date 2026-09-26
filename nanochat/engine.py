"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import ast
import operator
import signal
import warnings
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_init
from nanochat.lcqat.packing import pack_nibbles, unpack_nibbles

# Restricted expression evaluator for the calculator tool: parses the formula
# as an AST and only executes whitelisted nodes, so model-generated formulas can
# never reach arbitrary code execution (replaces a direct eval() call).
_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}
_SAFE_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_eval_ast(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "count"
        and not node.keywords
        and len(node.args) == 1
    ):
        target = _eval_ast(node.func.value)
        needle = _eval_ast(node.args[0])
        if isinstance(target, str) and isinstance(needle, str):
            return target.count(needle)
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def safe_eval_expression(formula: str):
    """Evaluate an arithmetic/string.count() expression from an untrusted source.

    Returns the value, or raises on anything outside the whitelist (callers
    treat rejection as a normal "calculator declined" outcome).
    """
    return _eval_ast(ast.parse(formula, mode="eval").body)


@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return safe_eval_expression(formula)
    except Exception:
        signal.alarm(0)
        # it's ok to ignore wrong calculator usage (untrusted model output)
        return None


def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    )
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    ]
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if ".count(" not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)


class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    quantized = False  # dispatch tag: gpt.py branches on this for attention

    def __init__(
        self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype
    ):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert (
            self.n_layers == other.n_layers
            and self.n_heads == other.n_heads
            and self.head_dim == other.head_dim
        )
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(
                self.batch_size, -1, -1
            ).clone()


class QuantizedKVCache:
    """4-bit packed KV cache (LC-QAT PRD section 7.1).

    Stores K/V as nibble-packed uint8 codebook indices instead of bf16 values:
    index buffers are [n_layers, B, T, H, ceil(D/2)] (two values per byte along
    D), resolved through per-(layer, head) FP32 codebooks of K <= 15 levels.
    Position bookkeeping matches KVCache (cache_seqlens int32 per batch
    element): write() stores at cache_seqlens, advance() moves past the step,
    mirroring FA3's append-at-cache_seqlens contract. FA3 cannot consume this
    layout - it pairs with the index-native dispatch_quant_attn op.
    """

    quantized = True  # dispatch tag: gpt.py branches on this for attention

    def __init__(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        num_layers,
        device,
        k_codebooks,
        v_codebooks,
    ):
        for label, cb in (("k_codebooks", k_codebooks), ("v_codebooks", v_codebooks)):
            if cb.shape != (num_layers, num_heads, cb.shape[-1]):
                raise ValueError(
                    f"{label} must be [n_layers, num_heads, K] = "
                    f"[{num_layers}, {num_heads}, K], got {tuple(cb.shape)}"
                )
            if cb.shape[-1] < 3 or cb.shape[-1] > 15 or cb.shape[-1] % 2 != 1:
                raise ValueError(
                    f"{label} last dim must be an odd K in [3, 15] (nibble-packed), "
                    f"got {cb.shape[-1]}"
                )
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.n_bytes = (head_dim + 1) // 2
        self.k_codebooks = k_codebooks.to(device=device, dtype=torch.float32)
        self.v_codebooks = v_codebooks.to(device=device, dtype=torch.float32)
        shape = (num_layers, batch_size, seq_len, num_heads, self.n_bytes)
        self.k_idx = torch.zeros(shape, dtype=torch.uint8, device=device)
        self.v_idx = torch.zeros(shape, dtype=torch.uint8, device=device)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding = None

    @staticmethod
    def storage_bytes(
        batch_size: int, num_heads: int, seq_len: int, head_dim: int, num_layers: int
    ) -> int:
        """Bytes of packed K+V index storage (uint8, nibble-packed along D)."""
        return 2 * num_layers * batch_size * seq_len * num_heads * ((head_dim + 1) // 2)

    @staticmethod
    def _quantize(x: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
        """[B, T, H, D] float -> uint8 codebook indices via midpoint bucketize."""
        x_fp32 = x.detach().to(torch.float32)
        b, t, h, d = x_fp32.shape
        indices = torch.empty(b, t, h, d, dtype=torch.uint8, device=x.device)
        for head in range(h):
            cb = codebooks[head]
            midpoints = (cb[:-1] + cb[1:]) * 0.5
            indices[:, :, head] = torch.bucketize(
                x_fp32[:, :, head].contiguous(), midpoints
            )
        return indices

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Quantize and append this step's K/V at cache_seqlens (does not advance).

        Args:
            layer_idx: which layer's buffers/codebooks to write.
            k, v: [B, T_cur, H, D] post-rotary keys and post-ve-mix values.
        """
        if not 0 <= layer_idx < self.n_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")
        t_cur = k.shape[1]
        for label, x in (("k", k), ("v", v)):
            if (
                x.ndim != 4
                or x.shape[0] != self.batch_size
                or x.shape[2:] != (self.n_heads, self.head_dim)
            ):
                raise ValueError(
                    f"{label} must be [B, T_cur, H, D] = [{self.batch_size}, T, "
                    f"{self.n_heads}, {self.head_dim}], got {tuple(x.shape)}"
                )
        for b in range(self.batch_size):
            pos = int(self.cache_seqlens[b])
            if pos + t_cur > self.max_seq_len:
                raise ValueError(
                    f"write of {t_cur} tokens at position {pos} exceeds seq_len "
                    f"{self.max_seq_len} (batch element {b})"
                )
        k_packed = pack_nibbles(self._quantize(k, self.k_codebooks[layer_idx]))
        v_packed = pack_nibbles(self._quantize(v, self.v_codebooks[layer_idx]))
        for b in range(self.batch_size):
            pos = int(self.cache_seqlens[b])
            self.k_idx[layer_idx, b, pos : pos + t_cur] = k_packed[b]
            self.v_idx[layer_idx, b, pos : pos + t_cur] = v_packed[b]

    def get_layer_indices(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_idx, v_idx) views [B, T, H, n_bytes] for a specific layer."""
        return self.k_idx[layer_idx], self.v_idx[layer_idx]

    def dequant_window(
        self, layer_idx: int, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve packed rows [start, end) back to FP32 [B, end-start, H, D]."""
        if not 0 <= start < end <= self.max_seq_len:
            raise ValueError(f"window [{start}, {end}) outside [0, {self.max_seq_len})")
        heads = torch.arange(self.n_heads, device=self.k_idx.device).view(1, 1, -1, 1)
        out = []
        for packed, codebooks in (
            (self.k_idx[layer_idx, :, start:end], self.k_codebooks),
            (self.v_idx[layer_idx, :, start:end], self.v_codebooks),
        ):
            indices = unpack_nibbles(packed, self.head_dim)
            out.append(codebooks[layer_idx][heads, indices.long()])
        return out[0], out[1]

    def prefill(self, other: "QuantizedKVCache"):
        """Copy packed KV from another cache (batch=1 prefill -> N parallel rows)."""
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert (
            self.n_layers == other.n_layers
            and self.n_heads == other.n_heads
            and self.head_dim == other.head_dim
        )
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_idx[:, :, :other_pos] = other.k_idx[:, :, :other_pos]
        self.v_idx[:, :, :other_pos] = other.v_idx[:, :, :other_pos]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(
                self.batch_size, -1, -1
            ).clone()


def kv_codebooks_from_model(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Export per-(layer, head) K/V codebooks from the live out-quantizers.

    Reads each block's attn.c_k/c_v LCQATLinear out_quantizer (training
    quantizes with one codebook per projection, shared across heads); the
    PRD 7.1 storage layout is per (layer, head), so the module codebook is
    repeated across that layer's KV heads. Returns fp32 [n_layers, n_kv_head, K].
    """
    k_list, v_list, n_kv_head = [], [], None
    for block in model.transformer.h:
        n_kv_head = block.attn.n_kv_head
        k_list.append(block.attn.c_k.out_quantizer.get_codebook())
        v_list.append(block.attn.c_v.out_quantizer.get_codebook())

    def per_head(lst):
        return (
            torch.stack(lst, dim=0)[:, None, :].expand(-1, n_kv_head, -1).contiguous()
        )

    return per_head(k_list), per_head(v_list)


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = (
            current_tokens or []
        )  # Current token sequence for this row
        self.forced_tokens = deque()  # Queue of tokens to force inject
        self.in_python_block = False  # Whether we are inside a python block
        self.python_expr_tokens = []  # Tokens of the current python expression
        self.completed = False  # Whether this row has completed generation


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer  # needed for tool use

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42,
        quantized_kv=False,
    ):
        """Same as generate, but does single prefill and then clones the KV cache.

        quantized_kv=True selects the QuantizedKVCache decode path (4-bit
        packed K/V resolved through per-head codebooks, dispatch_quant_attn)
        instead of the default bf16/FA3 cache; FA3 stays the default
        until parity is proven (tests/test_lcqat_engine_runtime.py).
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), (
            "expecting list of ints"
        )
        device = self.model.get_device()
        # Allocate the KV cache in the compute dtype so it matches what the forward pass emits
        dtype = COMPUTE_DTYPE
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")  # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id()  # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        if quantized_kv:
            k_codebooks, v_codebooks = kv_codebooks_from_model(self.model)

        def make_cache(batch_size, seq_len):
            if quantized_kv:
                return QuantizedKVCache(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    k_codebooks=k_codebooks,
                    v_codebooks=v_codebooks,
                    **kv_model_kwargs,
                )
            return KVCache(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
                **kv_model_kwargs,
            )

        kv_cache_prefill = make_cache(batch_size=1, seq_len=len(tokens))
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (
            (len(tokens) + max_tokens)
            if max_tokens is not None
            else self.model.config.sequence_len
        )
        kv_cache_decode = make_cache(batch_size=num_samples, seq_len=kv_length_hint)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = []  # contains the next token id along each row
            token_masks = []  # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = (
                    len(state.forced_tokens) > 0
                )  # are there tokens waiting to be forced in deque?
                token_masks.append(
                    0 if is_forced else 1
                )  # mask is 0 if forced, 1 if sampled
                next_token = (
                    state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                )
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(
                1
            )
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[
                :, -1, :
            ]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time

    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode(
        "The chemical formula of water is", prepend=bos_token_id
    )
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(
        prompt_tokens, num_samples=1, **kwargs
    )  # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0]  # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
