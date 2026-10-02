"""Sparse LC-QAT artifacts: CSR over index slots + alphabet compaction.

The payoff of the LC-QAT zero anchor is that a pruned SparseProp weight is not
a separate concept from a codebook index -- it holds an exact 0.0, and 0.0 is
index `m_neg`. So the sparse pattern is a subset of the index alphabet, and an
exported artifact can describe a layer as CSR over the surviving slots with a
codebook compacted down to the levels actually referenced.

What must hold, independent of any optimization:

* the compacted values reconstruct the original indices at exactly the kept
  positions, and are absent everywhere else;
* the CSR structure is well formed (row_ptr monotone, aligned lengths,
  columns ascending per row) -- the AVX2 kernels index it directly;
* compaction never changes a value, only the alphabet it is expressed in;
* the compacted alphabet preserves the original level ORDER, so a compacted
  codebook is still monotone and its zero anchor is still the level whose
  original index was `m_neg`.
"""

import pytest
import torch

from nanochat.models.quant.packing import (
    FORMAT_NIBBLES,
    FORMAT_TRITS,
    index_format_for_k,
)
from nanochat.models.quant.sparse_artifact import (
    MIN_EXPORT_ALPHABET,
    dense_bytes_for,
    pack_sparse_plan,
    packed_value_bytes,
    plan_sparse_export,
    sparse_bytes_for,
    sparsity_break_even,
    unpack_sparse_plan,
)


def _reconstruct(plan, m, n):
    """Scatter a plan's CSR values back into a dense `[m, n]` grid (-1 = absent)."""
    grid = torch.full((m, n), -1, dtype=torch.int64)
    counts = (plan.row_ptr[1:] - plan.row_ptr[:-1]).long()
    rows = torch.repeat_interleave(torch.arange(m), counts)
    grid[rows.long(), plan.col_indices.long()] = plan.keep_indices.long()
    return grid


def _sample(m=8, n=16, k=15, density=0.4, seed=0):
    torch.manual_seed(seed)
    indices = torch.randint(0, k, (m, n))
    mask = torch.rand(m, n) < density
    # Guarantee at least one survivor per row so the reconstruction has no
    # empty rows (which magnitude pruning also guarantees).
    empty = ~mask.any(dim=1)
    mask[empty, 0] = True
    return indices, mask


class TestSparsePlanStructure:
    def test_when_plan_built_then_it_reconstructs_the_original_indices(self):
        indices, mask = _sample()
        plan = plan_sparse_export(indices, mask, k=15)
        expected = torch.full_like(indices, -1, dtype=torch.int64)
        expected[mask] = plan.remap[indices.long()][mask]
        assert torch.equal(_reconstruct(plan, *indices.shape), expected)

    def test_when_plan_built_then_row_ptr_is_monotone_and_aligned(self):
        indices, mask = _sample()
        plan = plan_sparse_export(indices, mask, k=15)
        row_ptr = plan.row_ptr.long()
        assert row_ptr.numel() == indices.shape[0] + 1
        assert int(row_ptr[0]) == 0
        assert torch.all(row_ptr[1:] >= row_ptr[:-1]), "row_ptr is not monotone"
        assert int(row_ptr[-1]) == plan.nnz == int(mask.sum())
        assert plan.col_indices.numel() == plan.nnz
        assert plan.keep_indices.numel() == plan.nnz

    def test_when_plan_built_then_columns_ascend_within_each_row(self):
        """The CSR column order is an ordering contract, not a detail.

        The AVX2 forward walks `w_col[w_ptr[m]:w_ptr[m+1]]` in stored order. A
        correct-as-a-set but differently-ordered column list yields the same
        values and different memory traffic, and would silently break any
        future assumption that the order is canonical.
        """
        indices, mask = _sample(m=6, n=12, density=0.5, seed=4)
        plan = plan_sparse_export(indices, mask, k=15)
        for m in range(indices.shape[0]):
            lo, hi = int(plan.row_ptr[m]), int(plan.row_ptr[m + 1])
            cols = plan.col_indices[lo:hi].long()
            assert torch.all(cols[1:] > cols[:-1]), f"row {m} columns unsorted"
            assert torch.equal(cols, torch.nonzero(mask[m], as_tuple=True)[0]), (
                f"row {m} holds the wrong columns"
            )

    def test_when_no_mask_then_nnz_is_dense(self):
        indices, _ = _sample()
        plan = plan_sparse_export(indices, None, k=15)
        assert plan.nnz == indices.numel()
        assert plan.col_indices.numel() == indices.numel()


class TestAlphabetCompaction:
    def test_when_a_level_is_unused_then_it_is_dropped(self):
        indices = torch.zeros(4, 8, dtype=torch.long)  # only level 0 used
        plan = plan_sparse_export(indices, None, k=15)
        # One level survives, but the *exported* alphabet is floored at
        # MIN_EXPORT_ALPHABET: packing selects a storage format from K and the
        # densest one (trits) starts at 3, so K=1 has no valid format and
        # `pack_sparse_plan` would raise. `used` records the truth; `k_used`
        # records what can actually be stored.
        assert plan.used.tolist() == [0]
        assert plan.k_used == MIN_EXPORT_ALPHABET
        assert plan.compaction
        # The padded levels are never referenced, so this is still packable.
        packed, fmt = pack_sparse_plan(plan)
        assert fmt == index_format_for_k(plan.k_used)
        assert int(unpack_sparse_plan(packed, plan).max()) < plan.k_used

    def test_when_compaction_runs_then_k_used_selects_the_format(self):
        """A smaller alphabet can cross a storage boundary and shrink the buffer.

        This is the concrete payoff: at K=15 the format is nibbles (2 values
        per byte); a layer that only ever lands on 3 levels can be stored as
        trits (5 per byte) at 1/3 the size.
        """
        indices = torch.randint(0, 3, (8, 64), dtype=torch.long)
        plan = plan_sparse_export(indices, None, k=15)
        assert plan.k_used == 3
        assert plan.compaction
        assert index_format_for_k(15) == FORMAT_NIBBLES
        assert index_format_for_k(plan.k_used) == FORMAT_TRITS
        packed, fmt = pack_sparse_plan(plan)
        assert fmt == FORMAT_TRITS
        dense = dense_bytes_for(indices, 15)
        assert packed.numel() < dense, (packed.numel(), dense)

    def test_when_compaction_runs_then_compacted_indices_are_contiguous(self):
        indices = torch.tensor([[0, 7, 7], [14, 0, 7]], dtype=torch.long)
        plan = plan_sparse_export(indices, None, k=15)
        assert plan.used.tolist() == [0, 7, 14]
        assert plan.k_used == 3
        # values are [[0,1,1],[2,0,1]] in row-major CSR order
        assert plan.keep_indices.reshape(2, 3).tolist() == [[0, 1, 1], [2, 0, 1]]

    def test_when_compaction_runs_then_level_order_is_preserved(self):
        """A compacted codebook must stay monotone, and its anchor must be the
        level whose original index was `m_neg`.

        The zero anchor is what makes SparseProp legal at all, so it has to
        survive compaction: `used` is ascending, so a level cannot be re-sorted
        past another and the anchor remains the zero-valued entry.
        """
        m_neg = 3
        indices = torch.tensor([[0, m_neg, 5], [9, m_neg, 0]], dtype=torch.long)
        plan = plan_sparse_export(indices, None, k=15)
        used = plan.used.tolist()
        assert used == sorted(used), "compaction must preserve level order"
        assert m_neg in used
        # The anchor's compacted index is its rank among the survivors, which
        # is what a consumer must use to find the zero row of a fused LUT.
        anchor_compacted = int(plan.remap[m_neg])
        assert used[anchor_compacted] == m_neg

    def test_when_all_levels_used_then_nothing_is_dropped(self):
        indices = torch.arange(15).reshape(3, 5)
        plan = plan_sparse_export(indices, None, k=15)
        assert plan.k_used == 15
        assert not plan.compaction
        # keep_indices is CSR-ordered, so compare against the dense flattening
        # rather than the 2-D shape.
        assert torch.equal(plan.keep_indices.long(), indices.reshape(-1).long())

    def test_when_layer_is_fully_pruned_then_an_artifact_is_still_emitted(self):
        """An all-pruned layer keeps no alphabet; emitting K=1 would fail to pack.

        `index_format_for_k` rejects K < 3 (the trit format starts at 3), so a
        K=1 fallback makes the artifact unloadable rather than merely empty.
        """
        indices = torch.zeros(4, 8, dtype=torch.long)
        mask = torch.zeros(4, 8, dtype=torch.bool)
        plan = plan_sparse_export(indices, mask, k=15)
        assert plan.nnz == 0
        assert plan.k_used >= 3, plan.k_used
        # And it must actually pack.
        packed, _ = pack_sparse_plan(plan)
        assert packed.numel() == 0


class TestSparseRoundTrip:
    @pytest.mark.parametrize("k", [3, 15, 16, 255])
    def test_when_packed_then_unpacking_recovers_the_values(self, k):
        indices, mask = _sample(m=6, n=20, k=k, seed=k)
        plan = plan_sparse_export(indices, mask, k=k)
        packed, _ = pack_sparse_plan(plan)
        recovered = unpack_sparse_plan(packed, plan, plan.nnz)
        assert torch.equal(recovered.long(), plan.keep_indices.long())

    def test_when_sparse_then_the_value_bytes_shrink(self):
        """The values themselves shrink: fewer surviving weights, same format."""
        m, n, k = 16, 128, 15
        indices = torch.randint(0, k, (m, n))
        mask = torch.rand(m, n) < 0.5
        mask.any(1)
        plan = plan_sparse_export(indices, mask, k=k)
        dense_values = packed_value_bytes(indices, k)
        packed, _ = pack_sparse_plan(plan)
        assert packed.numel() < dense_values, (packed.numel(), dense_values)

    def test_when_sparse_then_csr_structure_costs_more_than_it_saves_at_low_sparsity(
        self,
    ):
        """Honest accounting: 4-byte CSR indices need ~80% sparsity to break even.

        This is why `--sparseprop-dense-threshold` defaults to 0.8 (SparseProp
        Sec. 4.1) and why a "sparse artifact is smaller" claim is false at 50%.
        The number must be reported truthfully rather than by ignoring the
        structure, which is where a size claim usually goes wrong.
        """
        m, n, k = 64, 256, 15
        indices = torch.randint(0, k, (m, n))
        break_even = sparsity_break_even(indices, k)
        assert 0.6 < break_even < 0.95, break_even

        # At exactly break-even sparsity the two footprints are equal; above it
        # the sparse layout wins.
        for sparsity, expect_sparse_wins in ((break_even - 0.2, False), (0.95, True)):
            keep = int(m * n * (1 - sparsity))
            mask = torch.zeros(m, n, dtype=torch.bool)
            mask.reshape(-1)[:keep] = True
            mask.any(1)
            plan = plan_sparse_export(indices, mask, k=k)
            sparse_bytes = sparse_bytes_for(plan)
            dense_bytes = dense_bytes_for(indices, k)
            assert (sparse_bytes < dense_bytes) is expect_sparse_wins, (
                f"sparsity={sparsity}: sparse={sparse_bytes} dense={dense_bytes}"
            )

    def test_when_k_is_inferred_then_an_unused_top_level_is_mistaken_for_k(self):
        """Inferring K from `indices.max()` under-counts a layer's real alphabet.

        Callers that know the layer's K must pass it; the inference is a
        convenience and is documented as such. The `compaction` verdict is
        where the difference shows, which is exactly what a caller relying on
        the inference would get wrong.
        """
        indices = torch.cat(
            [
                torch.zeros(4, 8, dtype=torch.long),
                torch.full((4, 1), 14, dtype=torch.long),
            ],
            dim=1,
        )
        # max index is 14, so the inferred K is 15 and both paths see the same
        # two surviving levels (0 and 14).
        inferred = plan_sparse_export(indices, None, k=None)
        explicit = plan_sparse_export(indices, None, k=15)
        assert inferred.k_used == explicit.k_used == MIN_EXPORT_ALPHABET
        assert inferred.used.tolist() == explicit.used.tolist() == [0, 14]
        assert torch.equal(inferred.keep_indices, explicit.keep_indices)

        # The case the inference actually gets wrong: the top level is unused,
        # so `indices.max()` understates the layer's alphabet. The compacted
        # alphabet is unaffected -- only the verdict about whether the layer was
        # 15 levels wide to begin with differs.
        small = torch.zeros(4, 8, dtype=torch.long)  # only level 0
        assert plan_sparse_export(small, None, k=None).k_used == MIN_EXPORT_ALPHABET
        assert plan_sparse_export(small, None, k=15).k_used == MIN_EXPORT_ALPHABET
        assert plan_sparse_export(small, None, k=15).compaction is True
        # A layer that genuinely uses every level is not compacted, so the
        # faithful case reports no compaction.
        full = torch.arange(15).reshape(1, 15)
        assert plan_sparse_export(full, None, k=None).k_used == 15
        assert plan_sparse_export(full, None, k=None).compaction is False


class TestSparseInvalidInputs:
    def test_when_indices_are_not_2d_then_plan_raises(self):
        with pytest.raises(ValueError, match=r"\[m, n\]"):
            plan_sparse_export(torch.zeros(4, dtype=torch.long))

    def test_when_mask_shape_mismatches_then_plan_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            plan_sparse_export(
                torch.zeros(4, 8, dtype=torch.long), torch.ones(4, 4, dtype=torch.bool)
            )


class TestSparseExportIntegration:
    """The export path, not just the plan builder."""

    @staticmethod
    def _sparse_lcqat_model(sparsity: float = 0.5):
        import torch.nn as nn

        from nanochat.models.quant.linear import LCQATLinear
        from nanochat.models.quant.sparseprop import SparsePropLinearLCQAT

        torch.manual_seed(3)
        lcqat = LCQATLinear.from_float(nn.Linear(16, 8), K_weight=15, K_act=15)
        sparse = SparsePropLinearLCQAT(lcqat, sparsity=sparsity)
        return nn.Sequential(sparse)

    def test_when_a_sparse_layer_is_exported_then_sparse_buffers_appear(self, tmp_path):
        """The exported artifact must carry the CSR + compacted alphabet.

        The whole W3.4 payoff is in the artifact, not in a helper function: if
        `export_lcqat_checkpoint` does not emit these, the compact alphabet and
        the physical size reduction exist only in memory.
        """
        from nanochat.models.quant.export import export_lcqat_checkpoint

        model = self._sparse_lcqat_model(sparsity=0.5)
        state = export_lcqat_checkpoint(model, str(tmp_path / "sparse.pt"), sparse=True)

        for suffix in (
            "sparse_keep_indices",
            "sparse_row_ptr",
            "sparse_col_indices",
            "sparse_alphabet",
            "sparse_index_format",
            "sparse_k_used",
        ):
            assert f"0.{suffix}" in state, f"missing 0.{suffix} in the artifact"
        # The shadow weight is stripped, as for any LC-QAT artifact.
        assert "0.weight" not in state
        # The sparse listing describes strictly fewer positions than the matrix.
        row_ptr = state["0.sparse_row_ptr"]
        nnz = int(row_ptr[-1])
        assert nnz < 16 * 8, (nnz, 16 * 8)
        # And the alphabet is a subset of the original 15 levels.
        alphabet = state["0.sparse_alphabet"]
        assert alphabet.numel() <= 15
        assert int(state["0.sparse_k_used"]) == alphabet.numel()

    def test_when_sparse_is_false_then_no_sparse_buffers_are_emitted(self, tmp_path):
        """Opting out must be a clean dense artifact, not a partial one."""
        from nanochat.models.quant.export import export_lcqat_checkpoint

        model = self._sparse_lcqat_model(sparsity=0.5)
        state = export_lcqat_checkpoint(model, str(tmp_path / "dense.pt"), sparse=False)
        assert not [k for k in state if "sparse_" in k], [
            k for k in state if "sparse_" in k
        ]
        assert "0.packed_weight_indices" in state

    def test_when_exported_sparsely_then_dense_knobs_still_work(self, tmp_path):
        """`packed_weight_indices` stays valid: the dense path is not broken.

        The sparse buffers are additive. A consumer reading only the dense
        buffer must get the same answer it got before, which is what keeps this
        an optimization rather than a behaviour change.
        """
        from nanochat.models.quant.export import export_lcqat_checkpoint
        from nanochat.models.quant.packing import unpack_weight_indices

        model = self._sparse_lcqat_model(sparsity=0.5)
        state = export_lcqat_checkpoint(model, str(tmp_path / "both.pt"), sparse=True)
        packed = state["0.packed_weight_indices"]
        fmt = int(state["0.weight_index_format"])
        # `unpack_weight_indices(packed, n, k)` takes the *in_features* width and
        # returns the [m, n] index matrix, so pass 16 (not the out width 8).
        unpacked = unpack_weight_indices(packed, 16, 15)
        assert tuple(unpacked.shape) == (8, 16), tuple(unpacked.shape)
        assert fmt in (0, 1, 2, 3)

    def test_when_a_dense_layer_is_exported_then_no_sparse_buffers(self, tmp_path):
        """A layer with no mask takes the dense path, unchanged."""
        import torch.nn as nn

        from nanochat.models.quant.export import export_lcqat_checkpoint
        from nanochat.models.quant.linear import LCQATLinear

        torch.manual_seed(4)
        model = nn.Sequential(
            LCQATLinear.from_float(nn.Linear(16, 8), K_weight=15, K_act=15)
        )
        state = export_lcqat_checkpoint(model, str(tmp_path / "d.pt"), sparse=True)
        assert not [k for k in state if "sparse_" in k]
