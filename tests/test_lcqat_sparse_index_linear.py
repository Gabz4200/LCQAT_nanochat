"""Parity tests for the sparse index-linear oracle (W5).

The claim under test is the one a CSR kernel can actually violate: the sparse
listing must reproduce the *masked* dense result, where the mask says which
slots are structurally zero. A dense-only oracle cannot catch a listing that
skips a surviving entry or reads the wrong column, because it would be built
from the same listing.

`tests/test_lcqat_sparse_artifact.py` covers the plan/packing side. This file
covers the math, including the mask polarity (keep vs prune), which is the
easiest thing in this codebase to get backwards.
"""

import pytest
import torch

from nanochat.models.quant.sparse_artifact import (
    MIN_EXPORT_ALPHABET,
    pack_sparse_plan,
    plan_sparse_export,
    unpack_sparse_plan,
)
from nanochat.ops.references.sparse_linear_reference import (
    reference_masked_index_linear,
    reference_sparse_index_linear,
    validate_sparse_index_linear_inputs,
)


def make_case(
    seed: int = 0, t: int = 7, m: int = 6, n: int = 8, k: int = 15, keep_p=0.7
):
    """Build a layer with a random keep-mask, plus the CSR listing for it."""
    torch.manual_seed(seed)
    act_lut = torch.sort(torch.randn(k)).values
    weight_lut = torch.sort(torch.randn(k)).values
    act_indices = torch.randint(0, k, (t, n), dtype=torch.uint8)
    indices = torch.randint(0, k, (m, n))
    keep = torch.rand(m, n) < keep_p
    # A pruned slot carries LC-QAT's exact zero anchor, so it is an ordinary
    # codebook index rather than a special marker.
    indices[~keep] = 0
    plan = plan_sparse_export(indices, keep, k=k)
    packed, _ = pack_sparse_plan(plan)
    unpacked = unpack_sparse_plan(packed, plan)
    return {
        "act_lut": act_lut,
        "weight_lut": weight_lut,
        "act_indices": act_indices,
        "indices": indices,
        "keep": keep,
        "plan": plan,
        "unpacked": unpacked,
        "m": m,
        "n": n,
        "k": k,
    }


def run_sparse(case) -> torch.Tensor:
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


def run_masked(case) -> torch.Tensor:
    return reference_masked_index_linear(
        case["act_indices"],
        case["act_lut"],
        case["indices"],
        case["weight_lut"],
        ~case["keep"],
        case["n"],
    )


class TestOracleAgreement:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_when_sparse_and_masked_oracles_run_then_they_agree(
        self, seed: int
    ) -> None:
        case = make_case(seed)
        torch.testing.assert_close(run_sparse(case), run_masked(case))

    @pytest.mark.parametrize("keep_p", [1.0, 0.5, 0.1, 0.0])
    def test_when_the_sparsity_is_extreme_then_the_oracles_still_agree(
        self, keep_p: float
    ) -> None:
        case = make_case(0, keep_p=keep_p)
        torch.testing.assert_close(run_sparse(case), run_masked(case))

    def test_when_the_layer_is_dense_then_nnz_equals_m_times_n(self) -> None:
        case = make_case(0, keep_p=1.0)
        assert case["plan"].nnz == case["m"] * case["n"]

    def test_when_the_layer_is_fully_pruned_then_the_output_is_all_zero(self) -> None:
        case = make_case(0, keep_p=0.0)
        assert case["plan"].nnz == 0
        out = run_sparse(case)
        assert out.shape == (case["act_indices"].shape[0], case["m"])
        assert torch.equal(out, torch.zeros_like(out))


class TestSparseSemantics:
    def test_when_a_row_is_fully_pruned_then_that_output_row_is_zero(self) -> None:
        """A wholly-pruned output row must contribute nothing, not garbage.

        The `continue` on an empty CSR window is the branch that makes this
        true; without it an unmasked gather would fill the row from stale
        buffers.
        """
        torch.manual_seed(2)
        m, n, k, t = 4, 6, 15, 3
        act_lut = torch.sort(torch.randn(k)).values
        weight_lut = torch.sort(torch.randn(k)).values
        act_indices = torch.randint(0, k, (t, n), dtype=torch.uint8)
        indices = torch.randint(0, k, (m, n))
        keep = torch.ones(m, n, dtype=torch.bool)
        keep[1] = False  # row 1 is entirely pruned
        indices[~keep] = 0
        plan = plan_sparse_export(indices, keep, k=k)
        packed, _ = pack_sparse_plan(plan)
        out = reference_sparse_index_linear(
            act_indices,
            act_lut,
            plan.row_ptr,
            plan.col_indices,
            unpack_sparse_plan(packed, plan),
            weight_lut[plan.used],
            m,
            n,
        )
        assert torch.equal(out[:, 1], torch.zeros(t))
        assert not torch.equal(out[:, 0], torch.zeros(t))

    def test_when_the_listing_is_read_then_the_pruned_slots_are_absent(self) -> None:
        """Every stored column must be a kept slot; none may be a pruned one."""
        case = make_case(1)
        plan = case["plan"]
        for row in range(case["m"]):
            start, stop = int(plan.row_ptr[row]), int(plan.row_ptr[row + 1])
            cols = plan.col_indices[start:stop].long().tolist()
            assert cols == sorted(cols), "CSR columns must ascend within a row"
            assert all(case["keep"][row, c].item() for c in cols)
            assert len(cols) == int(case["keep"][row].sum())

    def test_when_reconstructed_then_the_weights_match_the_masked_dense_matrix(
        self,
    ) -> None:
        """The listing must spell out exactly the masked weight matrix."""
        case = make_case(3)
        plan = case["plan"]
        values = case["weight_lut"][plan.used][case["unpacked"].long()]
        dense = torch.zeros(case["m"], case["n"])
        for row in range(case["m"]):
            start, stop = int(plan.row_ptr[row]), int(plan.row_ptr[row + 1])
            cols = plan.col_indices[start:stop].long()
            dense[row, cols] = values[start:stop]
        expected = case["weight_lut"][case["indices"].long()].masked_fill(
            ~case["keep"], 0.0
        )
        torch.testing.assert_close(dense, expected)

    def test_when_the_packed_values_are_unpacked_then_nnz_is_exact(self) -> None:
        """A nibble-packed buffer must not truncate the value list.

        The buffer's byte count is not its value count, so the count has to
        come from the plan; inferring it from the buffer silently drops every
        other value.
        """
        for k, keep_p in [(15, 0.7), (15, 0.3), (9, 0.5), (3, 0.6)]:
            case = make_case(0, keep_p=keep_p, k=k)
            assert case["unpacked"].numel() == case["plan"].nnz

    @pytest.mark.parametrize("k", [3, 5, 9, 15, 16])
    def test_when_the_alphabet_is_compacted_then_k_is_always_representable(
        self, k: int
    ) -> None:
        """Compaction may under-report, but never below a storable K.

        `index_format_for_k` rejects K < 3, so a two-level surviving alphabet
        has to be padded rather than exported as-is.
        """
        for keep_p in (1.0, 0.5, 0.1, 0.0):
            case = make_case(0, keep_p=keep_p, k=k)
            assert case["plan"].k_used >= MIN_EXPORT_ALPHABET
            # And the padding levels are never referenced. An empty listing has
            # no levels to check.
            if case["unpacked"].numel():
                assert int(case["unpacked"].max()) < case["plan"].k_used


class TestSparseValidation:
    def test_when_row_ptr_has_the_wrong_length_then_validation_raises(self) -> None:
        case = make_case(0)
        with pytest.raises(ValueError, match="row_ptr must have m\\+1"):
            validate_sparse_index_linear_inputs(
                case["act_indices"],
                case["act_lut"],
                torch.zeros(case["m"], dtype=torch.int64),
                case["plan"].col_indices,
                case["unpacked"],
                case["weight_lut"][case["plan"].used],
                case["m"],
                case["n"],
            )

    def test_when_col_indices_and_indices_disagree_then_validation_raises(self) -> None:
        case = make_case(0)
        with pytest.raises(ValueError, match="same length"):
            validate_sparse_index_linear_inputs(
                case["act_indices"],
                case["act_lut"],
                case["plan"].row_ptr,
                case["plan"].col_indices,
                case["unpacked"][:-1],
                case["weight_lut"][case["plan"].used],
                case["m"],
                case["n"],
            )

    def test_when_act_indices_are_not_uint8_then_validation_raises(self) -> None:
        """The dense reference demands uint8; the sparse path keeps that rule."""
        case = make_case(0)
        with pytest.raises(ValueError, match="uint8"):
            validate_sparse_index_linear_inputs(
                case["act_indices"].long(),
                case["act_lut"],
                case["plan"].row_ptr,
                case["plan"].col_indices,
                case["unpacked"],
                case["weight_lut"][case["plan"].used],
                case["m"],
                case["n"],
            )

    def test_when_the_masked_oracle_gets_a_packed_buffer_then_it_raises(self) -> None:
        """Guards the unpacked/packed API split.

        The masked oracle takes an already-unpacked index matrix. Accepting a
        packed buffer would mean silently re-interpreting it, which is how the
        two oracles first disagreed.
        """
        case = make_case(0)
        with pytest.raises(ValueError, match="unpacked"):
            reference_masked_index_linear(
                case["act_indices"],
                case["act_lut"],
                case["unpacked"],  # 1-D and packed, not a [m, n] matrix
                case["weight_lut"],
                ~case["keep"],
                case["n"],
            )

    def test_when_the_mask_shape_differs_from_the_weight_matrix_then_it_raises(
        self,
    ) -> None:
        case = make_case(0)
        with pytest.raises(ValueError, match="mask shape"):
            reference_masked_index_linear(
                case["act_indices"],
                case["act_lut"],
                case["indices"],
                case["weight_lut"],
                ~case["keep"][:-1],
                case["n"],
            )
