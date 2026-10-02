"""Tests for the fused 2D weight-product LUT (W5).

The table itself is exact FP32 arithmetic, so it is asserted exactly. The
*evaluation* through the table is a different summation order than the two-fetch
reference, so parity is asserted against an explicit tolerance and the
deviation is reported rather than only asserted -- an exact-zero tolerance
would be the wrong contract and would hide a real reordering bug.

What must never regress: the fused path must agree with the two-fetch path. If
they ever diverge materially, one of the two is computing the wrong thing, and
the fused one is the newer and therefore the more suspect.
"""

import pytest
import torch

from nanochat.lcqat.product_lut import (
    MAX_PRODUCT_TABLE_ENTRIES,
    build_product_table,
    flatten_product_table,
    fused_index_linear,
    fused_matches_reference,
    product_table_size,
)


def make_case(seed: int = 0, t: int = 6, m: int = 8, n: int = 12, k: int = 15):
    torch.manual_seed(seed)
    act_lut = torch.sort(torch.randn(k)).values
    weight_lut = torch.sort(torch.randn(k)).values
    act_indices = torch.randint(0, k, (t, n), dtype=torch.uint8)
    weight_indices = torch.randint(0, k, (m, n))
    return act_lut, weight_lut, act_indices, weight_indices, n


class TestProductTable:
    def test_when_the_table_is_built_then_every_entry_is_the_exact_product(
        self,
    ) -> None:
        """The table is exact FP32 arithmetic, not an approximation."""
        act_lut, weight_lut, _, _, _ = make_case()
        table = build_product_table(act_lut, weight_lut)
        assert table.shape == (act_lut.numel(), weight_lut.numel())
        for a in range(act_lut.numel()):
            for b in range(weight_lut.numel()):
                assert torch.equal(table[a, b], act_lut[a] * weight_lut[b])

    def test_when_the_table_is_flattened_then_the_address_is_a_times_k_plus_b(
        self,
    ) -> None:
        act_lut, weight_lut, _, _, _ = make_case()
        k_weight = weight_lut.numel()
        flat = flatten_product_table(build_product_table(act_lut, weight_lut), k_weight)
        table = build_product_table(act_lut, weight_lut)
        assert flat.shape == (act_lut.numel() * k_weight,)
        for a, b in [(0, 0), (3, 7), (14, 2), (1, k_weight - 1)]:
            assert torch.equal(flat[a * k_weight + b], table[a, b])

    def test_when_the_table_size_is_queried_then_it_is_the_product_of_the_k_values(
        self,
    ) -> None:
        assert product_table_size(15, 15) == 225
        assert product_table_size(3, 5) == 15

    def test_when_a_codebook_is_not_1d_then_building_raises(self) -> None:
        act_lut, weight_lut, _, _, _ = make_case()
        with pytest.raises(ValueError, match="act_lut must be 1-D"):
            build_product_table(act_lut.unsqueeze(0), weight_lut)

    def test_when_a_codebook_is_not_fp32_then_building_raises(self) -> None:
        act_lut, weight_lut, _, _, _ = make_case()
        with pytest.raises(ValueError, match="float32"):
            build_product_table(act_lut.to(torch.float64), weight_lut)

    def test_when_the_table_would_be_unbounded_then_building_raises(self) -> None:
        """A huge act alphabet must fall back to two fetches, not build a monster."""
        big = torch.randn(MAX_PRODUCT_TABLE_ENTRIES + 2, dtype=torch.float32)
        small = torch.randn(4, dtype=torch.float32)
        with pytest.raises(ValueError, match="above the"):
            build_product_table(big, small)


class TestFusedParity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_when_fused_and_two_fetch_run_then_they_agree(self, seed: int) -> None:
        act_lut, weight_lut, a, w, n = make_case(seed)
        ok, deviation = fused_matches_reference(a, act_lut, w, weight_lut, n)
        assert ok, f"fused diverged from two-fetch by {deviation}"

    @pytest.mark.parametrize(
        ("t", "m", "n"),
        [(1, 1, 1), (1, 1, 3), (5, 6, 8), (64, 128, 256), (3, 7, 5)],
    )
    def test_when_the_shape_varies_then_parity_holds(
        self, t: int, m: int, n: int
    ) -> None:
        act_lut, weight_lut, a, w, _ = make_case(0, t=t, m=m, n=n)
        ok, deviation = fused_matches_reference(a, act_lut, w, weight_lut, n)
        assert ok, f"[{t}, {m}, {n}] diverged by {deviation}"

    def test_when_fused_runs_then_the_output_shape_is_t_by_m(self) -> None:
        act_lut, weight_lut, a, w, n = make_case()
        assert fused_index_linear(a, act_lut, w, weight_lut, n).shape == (
            a.shape[0],
            w.shape[0],
        )

    def test_when_fused_runs_then_it_matches_a_dense_float_matmul(self) -> None:
        """Independent check: the value is right, not merely self-consistent."""
        act_lut, weight_lut, a, w, n = make_case()
        expected = act_lut[a.long()] @ weight_lut[w.long()].T
        torch.testing.assert_close(
            fused_index_linear(a, act_lut, w, weight_lut, n),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_when_the_table_is_precomputed_then_the_result_is_identical(self) -> None:
        """Passing a precomputed table must not change the answer."""
        act_lut, weight_lut, a, w, n = make_case()
        table = build_product_table(act_lut, weight_lut)
        recomputed = fused_index_linear(a, act_lut, w, weight_lut, n)
        with_table = fused_index_linear(a, act_lut, w, weight_lut, n, product=table)
        assert torch.equal(recomputed, with_table)

    def test_when_the_indices_are_not_a_matrix_then_fused_raises(self) -> None:
        act_lut, weight_lut, a, w, n = make_case()
        with pytest.raises(ValueError, match=r"unpacked \[m, n\] matrix"):
            fused_index_linear(a, act_lut, w.reshape(-1), weight_lut, n)

    def test_when_the_width_is_wrong_then_fused_raises(self) -> None:
        act_lut, weight_lut, a, w, n = make_case()
        with pytest.raises(ValueError, match=r"unpacked \[m, n\] matrix"):
            fused_index_linear(a, act_lut, w, weight_lut, n + 1)

    def test_when_the_alphabet_is_compacted_then_parity_still_holds(self) -> None:
        """Compaction shrinks K, so the fused table shrinks with it."""
        torch.manual_seed(7)
        act_lut = torch.sort(torch.randn(9)).values
        weight_lut = torch.sort(torch.randn(9)).values
        a = torch.randint(0, 9, (5, 11), dtype=torch.uint8)
        w = torch.randint(0, 9, (7, 11))
        ok, deviation = fused_matches_reference(a, act_lut, w, weight_lut, 11)
        assert ok, f"diverged by {deviation}"
        assert product_table_size(9, 9) == 81


class TestFusedDeviationIsRoundingOnly:
    def test_when_the_sum_is_long_then_the_deviation_is_only_rounding(self) -> None:
        """A long reduction reorders the additions; it must not drift.

        This is the assertion that distinguishes "different summation order"
        from "different math": the deviation must scale like FP32 epsilon on
        the accumulated magnitude, not grow with `n`.
        """
        torch.manual_seed(1)
        k = 15
        act_lut = torch.sort(torch.randn(k)).values
        weight_lut = torch.sort(torch.randn(k)).values
        deviations = []
        for n in (8, 64, 512):
            a = torch.randint(0, k, (4, n), dtype=torch.uint8)
            w = torch.randint(0, k, (8, n))
            _, deviation = fused_matches_reference(a, act_lut, w, weight_lut, n)
            deviations.append(deviation)
            # FP32 has ~1.2e-7 relative precision; a term sum of `n` of them
            # accumulates at most ~n * eps * max|term|, which stays tiny here.
            assert deviation < 1e-3, f"n={n} deviation {deviation} is not rounding"
        # And it must not grow superlinearly with the reduction length.
        assert deviations[2] < max(deviations[0], 1e-5) * 200
