"""Compiled-vs-oracle parity for the CSR sparse index-linear kernel (W5).

The kernel walks a `(row_ptr, col_indices, alphabet)` listing and the oracle in
`sparse_linear_reference` resolves the same listing through codebook indices.
They are two implementations of one statement, so this file is where a CSR
bookkeeping error becomes visible.

Zero-skip is the kernel-specific behaviour worth its own tests. LC-QAT's zero
anchor makes `0.0` an ordinary weight and magnitude pruning *stores* a pruned
slot as a real entry rather than removing it, so the `w == 0.0` early-out runs
in normal operation. It has to be observably equal to accumulating the zero,
and it has to stay inside the branch rather than corrupting the running sum.

The kernel needs a C++ toolchain (it is a JIT extension like the other ops).
Tests that only exercise validation skip when the extension cannot build.
"""

import pytest
import torch

from nanochat.models.quant.sparse_artifact import (
    pack_sparse_plan,
    plan_sparse_export,
    unpack_sparse_plan,
)
from nanochat.ops.references.sparse_linear_reference import (
    reference_sparse_index_linear,
)


def kernel_available() -> bool:
    """True when the C++ extension can be built, so JIT is not a test failure."""
    try:
        from nanochat.ops.kernels.cpu_loader import load_cpu_sparseprop_extension

        load_cpu_sparseprop_extension()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def sparse_kernel():
    """Require the compiled sparse kernel, failing rather than skipping.

    A `skipif` guard here would be actively harmful: a kernel that stops
    compiling makes every test in this file skip, and pytest calls that green.
    `AGENTS.md` already requires a compiler for the ops tests, so a load failure
    is a genuine failure, not an environment note.
    """
    assert kernel_available(), (
        "the CPU sparseprop extension must load; a skip here would hide a "
        "regression in the kernel behind a green run"
    )
    assert hasattr(torch.ops.nanochat, "lcqat_sparse_index_linear")
    return torch.ops.nanochat


def build_case(
    seed: int = 0, t: int = 6, m: int = 6, n: int = 8, k: int = 15, keep_p=0.7
):
    """A randomized layer plus the CSR listing and its resolved FP32 values.

    The weight codebook gets LC-QAT's exact zero anchor at a known level, so
    pruned slots really do land on `0.0` and the zero-skip branch is exercised
    the way it is in a real artifact. A random codebook has no exact zero, which
    would silently leave the branch untested.
    """
    torch.manual_seed(seed)
    act_lut = torch.sort(torch.randn(k)).values
    weight_lut = torch.sort(torch.randn(k)).values
    weight_lut[k // 2] = 0.0  # the zero anchor
    act_indices = torch.randint(0, k, (t, n), dtype=torch.uint8)
    indices = torch.randint(0, k, (m, n))
    keep = torch.rand(m, n) < keep_p
    # A pruned slot stores the zero anchor, which is what makes it skip.
    indices[~keep] = k // 2
    plan = plan_sparse_export(indices, keep, k=k)
    packed, _ = pack_sparse_plan(plan)
    unpacked = unpack_sparse_plan(packed, plan)
    alphabet = weight_lut[plan.used][unpacked.long()].to(torch.float32)
    return {
        "act_lut": act_lut,
        "weight_lut": weight_lut,
        "act_indices": act_indices,
        "indices": indices,
        "keep": keep,
        "plan": plan,
        "unpacked": unpacked,
        "alphabet": alphabet,
        "m": m,
        "n": n,
        "zero_level": k // 2,
    }


def run_kernel(case) -> torch.Tensor:
    from nanochat.ops.sparse_index_linear import sparse_index_linear_cpu

    return sparse_index_linear_cpu(
        case["act_indices"],
        case["act_lut"],
        case["plan"].col_indices.to(torch.int32),
        case["plan"].row_ptr.to(torch.int32),
        case["alphabet"],
        case["m"],
        case["n"],
    )


def run_oracle(case) -> torch.Tensor:
    return reference_sparse_index_linear(
        case["act_indices"],
        case["act_lut"],
        case["plan"].row_ptr,
        case["plan"].col_indices,
        case["unpacked"],
        case["weight_lut"][case["plan"].used],
        case["m"],
        case["n"],
    )


class TestKernelMatchesOracle:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_when_the_kernel_runs_then_it_matches_the_oracle(
        self, sparse_kernel, seed: int
    ) -> None:
        case = build_case(seed)
        torch.testing.assert_close(
            run_kernel(case), run_oracle(case), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("keep_p", [1.0, 0.7, 0.2, 0.0])
    def test_when_the_sparsity_is_extreme_then_the_kernel_still_matches(
        self, sparse_kernel, keep_p: float
    ) -> None:
        case = build_case(0, keep_p=keep_p)
        torch.testing.assert_close(
            run_kernel(case), run_oracle(case), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("k", [3, 5, 9, 15, 16])
    def test_when_the_alphabet_size_varies_then_the_kernel_still_matches(
        self, sparse_kernel, k: int
    ) -> None:
        case = build_case(0, k=k)
        torch.testing.assert_close(
            run_kernel(case), run_oracle(case), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize(
        ("t", "m", "n"), [(1, 1, 1), (1, 4, 3), (8, 1, 16), (32, 64, 128)]
    )
    def test_when_the_shape_varies_then_the_kernel_still_matches(
        self, sparse_kernel, t: int, m: int, n: int
    ) -> None:
        case = build_case(0, t=t, m=m, n=n)
        torch.testing.assert_close(
            run_kernel(case), run_oracle(case), rtol=1e-5, atol=1e-5
        )

    def test_when_the_kernel_runs_then_the_output_shape_is_t_by_m(
        self, sparse_kernel
    ) -> None:
        case = build_case(0, t=5, m=7)
        assert run_kernel(case).shape == (5, 7)


class TestZeroSkip:
    """The zero-skip branch, and what it can and cannot be tested for.

    Zero-skip is a *pure optimization*: skipping `w == 0.0` and accumulating it
    produce the same FP32 sum, so no value-level test can distinguish "the skip
    ran" from "the skip was deleted". Deleting the branch outright leaves every
    other assertion in this file green -- verified by reverting it and re-running.

    So these tests do not claim to cover the branch. They cover what is
    actually at stake: that the *listing* may contain zero-valued slots at all,
    that they never change the result, and that a whole zeroed row stays local
    to itself. A regression that corrupted the running sum when it met a zero
    would be caught here even though a missing branch would not.
    """

    def test_when_a_pruned_slot_is_stored_then_it_carries_the_exact_zero_anchor(
        self, sparse_kernel
    ) -> None:
        """The premise: pruned slots are stored entries with value 0.0.

        Without this, there is nothing for the skip to encounter and the whole
        question is vacuous.
        """
        case = build_case(0, keep_p=0.6)
        pruned = int((~case["keep"]).sum())
        assert pruned > 0, "need some pruned slots"
        assert case["plan"].nnz == case["keep"].sum(), (
            "pruned slots are stored, not removed from the listing"
        )
        # The stored values include exact zeros, since pruned slots resolve to
        # the zero anchor.
        assert int((case["alphabet"] == 0.0).sum()) > 0

    def test_when_the_listing_contains_zeros_then_the_result_matches_the_oracle(
        self, sparse_kernel
    ) -> None:
        """Zero-valued slots must not perturb the accumulation."""
        case = build_case(0, keep_p=0.6)
        assert int((case["alphabet"] == 0.0).sum()) > 0, "no zero slots to test"
        torch.testing.assert_close(
            run_kernel(case), run_oracle(case), rtol=1e-5, atol=1e-5
        )

    def test_when_every_stored_value_is_zero_then_the_output_is_zero(
        self, sparse_kernel
    ) -> None:
        case = build_case(0)
        case["alphabet"].zero_()
        out = run_kernel(case)
        assert torch.equal(out, torch.zeros_like(out))

    def test_when_a_whole_row_is_zeroed_then_only_that_row_disappears(
        self, sparse_kernel
    ) -> None:
        """Zero handling must stay inside its row, not clear the output.

        An early-exit bug that returned from the whole kernel rather than the
        inner loop would wipe every row; this catches that.
        """
        case = build_case(0)
        plan = case["plan"]
        start, stop = int(plan.row_ptr[1]), int(plan.row_ptr[2])
        before = run_kernel(case)
        case["alphabet"][start:stop].zero_()
        after = run_kernel(case)
        assert torch.equal(after[:, 1], torch.zeros(case["act_indices"].shape[0]))
        others = [i for i in range(case["m"]) if i != 1]
        assert torch.equal(after[:, others], before[:, others])


class TestKernelRejectsBadInputs:
    def test_when_the_backend_is_unknown_then_it_raises(self, sparse_kernel) -> None:
        """No dense fallback: a silent downgrade would defeat the point."""
        from nanochat.ops.sparse_index_linear import sparse_index_linear

        case = build_case(0)
        with pytest.raises(ValueError, match="no fallback"):
            sparse_index_linear(
                case["act_indices"],
                case["act_lut"],
                case["plan"].col_indices.to(torch.int32),
                case["plan"].row_ptr.to(torch.int32),
                case["alphabet"],
                case["m"],
                case["n"],
                backend="cuda",
            )

    def test_when_the_columns_are_int64_then_the_compiled_path_raises(
        self, sparse_kernel
    ) -> None:
        """The kernel reads int32; silently casting would hide a caller bug."""
        from nanochat.ops.sparse_index_linear import sparse_index_linear_cpu

        case = build_case(0)
        with pytest.raises(ValueError, match="int32"):
            sparse_index_linear_cpu(
                case["act_indices"],
                case["act_lut"],
                case["plan"].col_indices.to(torch.int64),
                case["plan"].row_ptr.to(torch.int32),
                case["alphabet"],
                case["m"],
                case["n"],
            )

    def test_when_nnz_disagrees_between_buffers_then_it_raises(
        self, sparse_kernel
    ) -> None:
        from nanochat.ops.sparse_index_linear import sparse_index_linear_cpu

        case = build_case(0)
        with pytest.raises(ValueError, match="same length"):
            sparse_index_linear_cpu(
                case["act_indices"],
                case["act_lut"],
                case["plan"].col_indices.to(torch.int32),
                case["plan"].row_ptr.to(torch.int32),
                case["alphabet"][:-1],
                case["m"],
                case["n"],
            )

    def test_when_the_activation_indices_are_not_uint8_then_it_raises(
        self, sparse_kernel
    ) -> None:
        from nanochat.ops.sparse_index_linear import sparse_index_linear_cpu

        case = build_case(0)
        with pytest.raises(ValueError, match="uint8"):
            sparse_index_linear_cpu(
                case["act_indices"].long(),
                case["act_lut"],
                case["plan"].col_indices.to(torch.int32),
                case["plan"].row_ptr.to(torch.int32),
                case["alphabet"],
                case["m"],
                case["n"],
            )
