"""
Tests for the SparseProp kernel stack on the training path:
three-way naive == cpu == gpu parity against an independent
pure-Python loop oracle, the dispatch error contract, the
listing layout contracts the GPU kernels depend on, structural
validation, torch.library opcheck, and the paper's Sec. 4.2
measured crossover (dense below 80% sparsity, measured above).

python -m pytest tests/test_sparseprop_gpu.py -v
"""

import pytest
import torch
import torch.nn as nn

from nanochat.models.quant.sparseprop import (
    SparsePropLinear,
    SparsePropLinearLCQAT,
    inject_sparseprop_layers,
)
from nanochat.ops import dispatch_sparseprop_backward, dispatch_sparseprop_forward
from nanochat.ops.kernels.gpu_loader import vulkan_available
from nanochat.ops.sparseprop import (
    build_csr_csc_from_mask,
    gather_values_from_dense,
    gather_values_from_dense_csc,
    sparseprop_backward_cpu,
    sparseprop_forward_cpu,
)

requires_vulkan = pytest.mark.skipif(
    not vulkan_available(), reason="Vulkan device unavailable for the Taichi backend"
)


@pytest.fixture
def tiny_model():
    """Depth-2 GPT, the shape the SparseProp integration tests use.

    `init_weights()` matters here: a zero `c_proj` transmits no
    gradient, and the codebook-gradient assertion below would
    pass vacuously.
    """
    from nanochat.models.backbone import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model = model.to_empty(device="cpu")
    model.init_weights()
    return model


def make_case(
    M: int = 6,
    K: int = 8,
    B: int = 5,
    sparsity: float = 0.5,
    seed: int = 0,
    bias: bool = True,
    dense_row: bool = False,
    dense_col: bool = False,
) -> dict:
    """One parity case: a weight, a keep-mask, and both listings.

    Every row and column keeps at least one entry (an empty row or
    column is legal but walks nothing), and optional fully-kept
    rows/columns exercise the adversarial end of the walk.
    """
    torch.manual_seed(seed)
    weight = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) < (1.0 - sparsity)
    for i in range(M):
        if not mask[i].any():
            mask[i, 0] = True
    for j in range(K):
        if not mask[:, j].any():
            mask[0, j] = True
    if dense_row:
        mask[0, :] = True
    if dense_col:
        mask[:, 0] = True
    x = torch.randn(B, K, dtype=torch.float32)
    bias_t = torch.randn(M, dtype=torch.float32) if bias else None
    w_ptr, w_col, w_cptr, w_row = build_csr_csc_from_mask(mask)
    return dict(
        x=x,
        weight_dense=weight,
        mask=mask,
        bias=bias_t,
        w_val=gather_values_from_dense(weight, w_col, w_ptr, M),
        w_col=w_col,
        w_ptr=w_ptr,
        w_val_csc=gather_values_from_dense_csc(weight, w_row, w_cptr),
        w_row=w_row,
        w_cptr=w_cptr,
        gY=torch.randn(M, B, dtype=torch.float32),
        M=M,
        K=K,
        B=B,
    )


def loop_oracle_forward(
    weight: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Pure-Python SpMM over the keep-mask, independent of every
    reference in the repo (the repo's own naive backend shares
    structure with the kernels under test)."""
    M, K = mask.shape
    B = x.shape[0]
    out = torch.zeros(M, B)
    for m in range(M):
        for b in range(B):
            acc = 0.0
            for k in range(K):
                if mask[m, k]:
                    acc += float(weight[m, k]) * float(x[b, k])
            if bias is not None:
                acc += float(bias[m])
            out[m, b] = acc
    return out


def loop_oracle_backward(
    gY: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python gradients: gX = W_masked.T @ gY, gW = gY @ x.T
    restricted to the keep-mask."""
    M, K = mask.shape
    B = x.shape[0]
    gX = torch.zeros(K, B)
    gW = torch.zeros(M, K)
    for k in range(K):
        for b in range(B):
            acc = 0.0
            for m in range(M):
                if mask[m, k]:
                    acc += float(weight[m, k]) * float(gY[m, b])
            gX[k, b] = acc
    for m in range(M):
        for k in range(K):
            if mask[m, k]:
                acc = 0.0
                for b in range(B):
                    acc += float(gY[m, b]) * float(x[b, k])
                gW[m, k] = acc
    return gX, gW


# ---------------------------------------------------------------------------
# Forward and backward parity: naive == cpu == gpu == loop oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("M", "K", "B", "sparsity"),
    [
        (6, 8, 5, 0.5),  # B % 8 != 0 (scalar tail in the AVX2 walk)
        (16, 32, 13, 0.75),  # default sparsity, misaligned batch
        (32, 64, 16, 0.9),  # high sparsity, the sparse regime
        (8, 8, 8, 0.5),  # square, vector-width aligned
    ],
    ids=["small", "default", "high", "aligned"],
)
def test_when_naive_backend_then_matches_independent_loop(
    M: int, K: int, B: int, sparsity: float
) -> None:
    case = make_case(M, K, B, sparsity, seed=M * 1000 + K)
    got = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["bias"],
        case["M"],
        backend="naive",
    )
    expected = loop_oracle_forward(
        case["weight_dense"], case["mask"], case["x"], case["bias"]
    )
    assert torch.allclose(got, expected, atol=1e-5)

    gX, gW_val = dispatch_sparseprop_backward(
        case["gY"],
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["w_val_csc"],
        case["w_row"],
        case["w_cptr"],
        case["M"],
        case["K"],
        backend="naive",
    )
    gX_ref, gW_ref = loop_oracle_backward(
        case["gY"], case["weight_dense"], case["mask"], case["x"]
    )
    assert torch.allclose(gX, gX_ref, atol=1e-5)
    # The naive backend returns the nnz listing: extract the same
    # positions from the loop oracle's dense gradient.
    row_idx = torch.repeat_interleave(
        torch.arange(M),
        (case["w_ptr"][1:] - case["w_ptr"][:-1]).to(torch.int64),
    )
    lin_idx = row_idx.long() * K + case["w_col"].long()
    assert torch.allclose(gW_val, gW_ref.reshape(-1)[lin_idx], atol=1e-5)


@pytest.mark.parametrize(
    ("M", "K", "B", "sparsity"),
    [
        (6, 8, 5, 0.5),
        (16, 32, 13, 0.75),
        (32, 64, 16, 0.9),
        (8, 8, 8, 0.5),
    ],
    ids=["small", "default", "high", "aligned"],
)
def test_when_cpu_backend_then_matches_naive(
    M: int, K: int, B: int, sparsity: float
) -> None:
    case = make_case(M, K, B, sparsity, seed=M * 1000 + K + 1)
    naive = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["bias"],
        case["M"],
        backend="naive",
    )
    cpu = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["bias"],
        case["M"],
        backend="cpu",
    )
    assert torch.allclose(cpu, naive, atol=1e-5)

    gX_n, gW_n = dispatch_sparseprop_backward(
        case["gY"],
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["w_val_csc"],
        case["w_row"],
        case["w_cptr"],
        case["M"],
        case["K"],
        backend="naive",
    )
    gX_c, gW_c = dispatch_sparseprop_backward(
        case["gY"],
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["w_val_csc"],
        case["w_row"],
        case["w_cptr"],
        case["M"],
        case["K"],
        backend="cpu",
    )
    assert torch.allclose(gX_c, gX_n, atol=1e-5)
    assert torch.allclose(gW_c, gW_n, atol=1e-5)


@requires_vulkan
@pytest.mark.parametrize(
    ("M", "K", "B", "sparsity"),
    [
        (6, 8, 5, 0.5),
        (16, 32, 13, 0.75),
        (32, 64, 16, 0.9),
        (8, 8, 8, 0.5),
    ],
    ids=["small", "default", "high", "aligned"],
)
def test_when_gpu_backend_then_matches_naive(
    M: int, K: int, B: int, sparsity: float
) -> None:
    case = make_case(M, K, B, sparsity, seed=M * 1000 + K + 2)
    naive = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["bias"],
        case["M"],
        backend="naive",
    )
    gpu = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["bias"],
        case["M"],
        backend="gpu",
    )
    assert gpu.shape == naive.shape
    assert torch.allclose(gpu, naive, atol=1e-5)

    gX_n, gW_n = dispatch_sparseprop_backward(
        case["gY"],
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["w_val_csc"],
        case["w_row"],
        case["w_cptr"],
        case["M"],
        case["K"],
        backend="naive",
    )
    gX_g, gW_g = dispatch_sparseprop_backward(
        case["gY"],
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["w_val_csc"],
        case["w_row"],
        case["w_cptr"],
        case["M"],
        case["K"],
        backend="gpu",
    )
    assert torch.allclose(gX_g, gX_n, atol=1e-5)
    assert torch.allclose(gW_g, gW_n, atol=1e-5)


@pytest.mark.parametrize(
    ("M", "K", "B", "sparsity"),
    [(16, 32, 13, 0.75), (32, 64, 16, 0.9)],
    ids=["default", "high"],
)
def test_when_no_bias_then_matches_naive(
    M: int, K: int, B: int, sparsity: float
) -> None:
    case = make_case(M, K, B, sparsity, seed=7, bias=False)
    assert case["bias"] is None
    naive = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        None,
        case["M"],
        backend="naive",
    )
    cpu = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        None,
        case["M"],
        backend="cpu",
    )
    assert torch.allclose(cpu, naive, atol=1e-5)


@pytest.mark.parametrize(
    ("dense_row", "dense_col"),
    [(True, False), (False, True), (True, True)],
    ids=["dense-row", "dense-col", "both"],
)
def test_when_adversarial_mask_then_cpu_matches_naive(
    dense_row: bool, dense_col: bool
) -> None:
    """A fully-kept row (or column) is the adversarial end of the
    CSR/CSC walk: one listing entry count at the maximum."""
    case = make_case(16, 32, 13, 0.75, seed=9, dense_row=dense_row, dense_col=dense_col)
    naive = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["bias"],
        case["M"],
        backend="naive",
    )
    cpu = dispatch_sparseprop_forward(
        case["x"].t(),
        case["weight_dense"],
        case["mask"],
        case["w_val"],
        case["w_col"],
        case["w_ptr"],
        case["bias"],
        case["M"],
        backend="cpu",
    )
    assert torch.allclose(cpu, naive, atol=1e-5)


def test_when_dispatch_unknown_backend_then_value_error() -> None:
    case = make_case(6, 8, 5, 0.5, seed=3)
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_sparseprop_forward(
            case["x"].t(),
            case["weight_dense"],
            case["mask"],
            case["w_val"],
            case["w_col"],
            case["w_ptr"],
            case["bias"],
            case["M"],
            backend="cuda9",
        )
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_sparseprop_backward(
            case["gY"],
            case["x"].t(),
            case["weight_dense"],
            case["mask"],
            case["w_val"],
            case["w_col"],
            case["w_ptr"],
            case["w_val_csc"],
            case["w_row"],
            case["w_cptr"],
            case["M"],
            case["K"],
            backend="cuda9",
        )


# ---------------------------------------------------------------------------
# Listing layout contracts (the GPU kernels walk the emitted order)
# ---------------------------------------------------------------------------


class TestListingLayoutContract:
    def test_when_structure_built_then_csr_cols_ascend_per_row(self) -> None:
        case = make_case(32, 64, 16, 0.75, seed=21)
        for m in range(case["M"]):
            cols = case["w_col"][
                int(case["w_ptr"][m]) : int(case["w_ptr"][m + 1])
            ].tolist()
            assert cols == sorted(cols)

    def test_when_structure_built_then_csc_rows_ascend_per_column(self) -> None:
        case = make_case(32, 64, 16, 0.75, seed=22)
        for k in range(case["K"]):
            rows = case["w_row"][
                int(case["w_cptr"][k]) : int(case["w_cptr"][k + 1])
            ].tolist()
            assert rows == sorted(rows)

    def test_when_structure_built_then_csr_and_csc_cover_same_nnz(self) -> None:
        case = make_case(32, 64, 16, 0.75, seed=23)
        assert case["w_col"].numel() == case["w_row"].numel()
        csr_pairs = set(
            zip(
                torch.repeat_interleave(
                    torch.arange(case["M"]),
                    (case["w_ptr"][1:] - case["w_ptr"][:-1]).to(torch.int64),
                ).tolist(),
                case["w_col"].tolist(),
            )
        )
        csc_pairs = set(
            zip(
                case["w_row"].tolist(),
                torch.repeat_interleave(
                    torch.arange(case["K"]),
                    (case["w_cptr"][1:] - case["w_cptr"][:-1]).to(torch.int64),
                ).tolist(),
            )
        )
        assert csr_pairs == csc_pairs


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_when_csr_index_out_of_range_then_value_error(self) -> None:
        case = make_case(16, 32, 13, 0.75, seed=31)
        bad = case["w_col"].clone()
        bad[0] = case["K"]
        with pytest.raises(ValueError, match="w_col"):
            sparseprop_forward_cpu(
                case["x"].t(),
                case["w_val"],
                bad,
                case["w_ptr"],
                case["bias"],
                case["M"],
            )

    def test_when_csc_index_out_of_range_then_value_error(self) -> None:
        case = make_case(16, 32, 13, 0.75, seed=32)
        bad = case["w_row"].clone()
        bad[0] = case["M"]
        with pytest.raises(ValueError, match="w_row"):
            sparseprop_backward_cpu(
                case["gY"],
                case["x"].t(),
                case["w_val"],
                case["w_col"],
                case["w_ptr"],
                case["w_val_csc"],
                bad,
                case["w_cptr"],
                case["M"],
                case["K"],
            )

    def test_when_pointers_not_monotone_then_value_error(self) -> None:
        case = make_case(16, 32, 13, 0.75, seed=33)
        bad = case["w_ptr"].clone()
        bad[2] = bad[1] - 1
        with pytest.raises(ValueError, match="monotone"):
            sparseprop_forward_cpu(
                case["x"].t(),
                case["w_val"],
                case["w_col"],
                bad,
                case["bias"],
                case["M"],
            )

    def test_when_pointers_do_not_end_at_nnz_then_value_error(self) -> None:
        case = make_case(16, 32, 13, 0.75, seed=34)
        bad = case["w_ptr"].clone()
        bad[-1] = bad[-1] - 1
        with pytest.raises(ValueError, match="nnz"):
            sparseprop_forward_cpu(
                case["x"].t(),
                case["w_val"],
                case["w_col"],
                bad,
                case["bias"],
                case["M"],
            )

    def test_when_pointers_do_not_start_at_zero_then_value_error(self) -> None:
        case = make_case(16, 32, 13, 0.75, seed=35)
        bad = case["w_cptr"].clone()
        bad[0] = 1
        with pytest.raises(ValueError, match="start at 0"):
            sparseprop_backward_cpu(
                case["gY"],
                case["x"].t(),
                case["w_val"],
                case["w_col"],
                case["w_ptr"],
                case["w_val_csc"],
                case["w_row"],
                bad,
                case["M"],
                case["K"],
            )

    def test_when_cpp_op_given_bad_listing_then_runtime_error(self) -> None:
        """The C++ kernel validates too: the op is reachable directly
        through torch.ops, so the check must live in the kernel, not
        only in the Python facade."""
        from nanochat.ops.sparseprop import _ensure_cpu_op

        _ensure_cpu_op()
        case = make_case(16, 32, 13, 0.75, seed=36)
        bad = case["w_col"].clone()
        bad[0] = case["K"]
        with pytest.raises(RuntimeError, match="out of range"):
            torch.ops.nanochat.lcqat_sparseprop_forward(
                case["w_val"].contiguous(),
                bad.contiguous(),
                case["w_ptr"].contiguous(),
                case["x"].t().contiguous(),
                case["bias"].contiguous(),
                int(case["M"]),
            )


# ---------------------------------------------------------------------------
# FakeTensor / opcheck (the forward fake's parameter order is schema order)
# ---------------------------------------------------------------------------


def test_when_forward_op_then_opcheck_passes() -> None:
    from nanochat.ops.sparseprop import _ensure_cpu_op

    _ensure_cpu_op()
    case = make_case(6, 8, 5, 0.5, seed=41)
    args = (
        case["w_val"].contiguous(),
        case["w_col"].contiguous(),
        case["w_ptr"].contiguous(),
        case["x"].t().contiguous(),
        case["bias"].contiguous(),
        int(case["M"]),
    )
    torch.library.opcheck(torch.ops.nanochat.lcqat_sparseprop_forward, args)


# ---------------------------------------------------------------------------
# The paper's Sec. 4.2 measured crossover on the training path
# ---------------------------------------------------------------------------


class TestMeasuredCrossover:
    def test_when_below_dense_threshold_then_dense_without_measuring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        lin = nn.Linear(16, 8, bias=True)
        layer = SparsePropLinear.from_linear(lin, sparsity=0.5)

        def _fail(self, weight: torch.Tensor, x: torch.Tensor) -> str:
            raise AssertionError("no measurement below the dense threshold")

        monkeypatch.setattr(SparsePropLinear, "_measure_kernel", _fail)
        choice = layer._resolve_kernel(layer.weight, torch.randn(2, 16))
        assert choice == "dense"

    def test_when_above_dense_threshold_then_kernel_is_measured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        lin = nn.Linear(16, 8, bias=True)
        layer = SparsePropLinear.from_linear(lin, sparsity=0.9)
        mask = torch.rand(8, 16) < 0.15
        layer._set_mask(mask)
        calls: list[int] = []
        orig = SparsePropLinear._measure_kernel

        def _counting(self, weight: torch.Tensor, x: torch.Tensor) -> str:
            calls.append(1)
            return orig(self, weight, x)

        monkeypatch.setattr(SparsePropLinear, "_measure_kernel", _counting)
        layer(torch.randn(2, 16))
        assert len(calls) == 1
        assert layer._kernel_choice in ("dense", "cpu", "gpu")

    def test_when_kernel_resolved_then_measurement_runs_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        layer = SparsePropLinear(16, 8, sparsity=0.9)
        layer._set_mask(torch.rand(8, 16) < 0.15)
        calls: list[int] = []
        orig = SparsePropLinear._measure_kernel

        def _counting(self, weight: torch.Tensor, x: torch.Tensor) -> str:
            calls.append(1)
            return orig(self, weight, x)

        monkeypatch.setattr(SparsePropLinear, "_measure_kernel", _counting)
        x = torch.randn(4, 16)
        layer(x)
        layer(x)
        layer(x)
        assert len(calls) == 1

    def test_when_mask_changes_then_choice_is_invalidated(self) -> None:
        torch.manual_seed(0)
        layer = SparsePropLinear(16, 8, sparsity=0.9)
        layer._set_mask(torch.rand(8, 16) < 0.15)
        layer._kernel_choice = "cpu"
        # A new prune event must re-resolve (and re-measure): the
        # pattern changed, so the old measurement no longer applies.
        layer._set_mask(torch.rand(8, 16) < 0.15)
        assert layer._kernel_choice is None

    @pytest.mark.parametrize("kernel", ["cpu"])
    def test_when_sparse_kernel_forced_then_gradients_match_dense(
        self, kernel: str
    ) -> None:
        """Correctness must not depend on which kernel the crossover
        picked: the sparse path is value-identical to the dense
        masked GEMM, with exact zeros at pruned positions."""
        torch.manual_seed(0)
        M, K = 8, 16
        lin = nn.Linear(K, M, bias=True)
        layer = SparsePropLinear.from_linear(lin, sparsity=0.9)
        mask = torch.rand(M, K) < 0.15
        layer._set_mask(mask)
        layer._kernel_choice = kernel

        x = torch.randn(4, K)
        x_sparse = x.clone().requires_grad_(True)
        y = layer(x_sparse)
        loss = y.square().mean()
        loss.backward()

        lin_ref = nn.Linear(K, M, bias=True)
        with torch.no_grad():
            lin_ref.weight.copy_(layer.weight)
            lin_ref.bias.copy_(layer.bias)
        x_ref = x.clone().requires_grad_(True)
        y_ref = lin_ref(x_ref)
        loss_ref = y_ref.square().mean()
        loss_ref.backward()

        assert torch.allclose(y, y_ref, atol=1e-5)
        assert torch.allclose(x_sparse.grad, x_ref.grad, atol=1e-5)
        assert torch.allclose(
            layer.weight.grad[mask], lin_ref.weight.grad[mask], atol=1e-5
        )
        # The freeze contract: pruned slots hold an exact zero
        # gradient (a mask multiply could only approximate this).
        assert bool((layer.weight.grad[~mask] == 0).all())

    @requires_vulkan
    def test_when_gpu_kernel_forced_then_gradients_match_dense(self) -> None:
        torch.manual_seed(0)
        M, K = 8, 16
        lin = nn.Linear(K, M, bias=True)
        layer = SparsePropLinear.from_linear(lin, sparsity=0.9)
        mask = torch.rand(M, K) < 0.15
        layer._set_mask(mask)
        layer._kernel_choice = "gpu"

        x = torch.randn(4, K)
        x_sparse = x.clone().requires_grad_(True)
        y = layer(x_sparse)
        loss = y.square().mean()
        loss.backward()

        lin_ref = nn.Linear(K, M, bias=True)
        with torch.no_grad():
            lin_ref.weight.copy_(layer.weight)
            lin_ref.bias.copy_(layer.bias)
        x_ref = x.clone().requires_grad_(True)
        y_ref = lin_ref(x_ref)
        loss_ref = y_ref.square().mean()
        loss_ref.backward()

        assert torch.allclose(y, y_ref, atol=1e-5)
        assert torch.allclose(x_sparse.grad, x_ref.grad, atol=1e-5)
        assert torch.allclose(
            layer.weight.grad[mask], lin_ref.weight.grad[mask], atol=1e-5
        )
        assert bool((layer.weight.grad[~mask] == 0).all())


class TestSparsePathLCQAT:
    def test_when_sparse_kernel_forced_then_codebook_gradients_flow(
        self, tiny_model
    ) -> None:
        """The sparse kernels carry the STE gradient into the codebook:
        forcing the sparse path must not break LC-QAT's learned
        codebook, and pruned weight slots stay frozen at exactly zero."""
        from nanochat.models.quant import LayerKConfig, retrofit_model

        config = LayerKConfig(min_linear_dim=1)
        model = retrofit_model(tiny_model, config)
        inject_sparseprop_layers(model, sparsity=0.5, with_lcqat=True)
        module = next(
            m for m in model.modules() if isinstance(m, SparsePropLinearLCQAT)
        )
        module._kernel_choice = "cpu"

        x = torch.randn(4, 8, 32, dtype=torch.float32)
        y = module(x)
        y.sum().backward()

        assert module.weight.grad is not None
        for name, p in module.weight_quantizer.named_parameters():
            assert p.grad is not None, f"codebook param {name} has no gradient"
        assert bool((module.weight.grad[~module.sparsity_mask] == 0).all()), (
            "pruned weight slots must hold exactly zero gradient"
        )
