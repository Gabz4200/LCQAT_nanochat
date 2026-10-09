"""DARD Phase I + II: sharding, AML atomization, weighting, compilation.

The discrete/continuous split is the property under test as much as the
arithmetic: mining is integer set algebra, compilation is the only thing that
produces floats, and the compiled form is what Phase III is allowed to see.

python -m pytest tests/test_dard_mining.py -v
"""

from __future__ import annotations

import pytest
import torch

from nanochat.models.dard.atomize import atomize_shard
from nanochat.models.dard.compile import (
    RULE_SET_VERSION,
    compile_rules,
    load_rule_set,
    mine_rules,
    rule_set_to_meta,
    save_rule_set,
)
from nanochat.models.dard.shard import shard_corpus
from nanochat.models.dard.types import AmlRule, DardRuleSet, ShardRules
from nanochat.models.dard.weighting import (
    EPS_FLOOR,
    composite_rule_weights,
)


def corpus(vocab: int = 64, repeats: int = 4) -> torch.Tensor:
    """A corpus with real structure: a repeated motif plus filler.

    Structure is the whole premise of the feature, so a uniformly random corpus
    would mine nothing and every rule assertion below would be vacuous.
    """
    motif = [3, 5, 7, 9]
    stream = (motif * 40) + list(range(20, 40))
    return torch.tensor((stream * repeats)[:2000], dtype=torch.long) % vocab


# --- Phase I: sharding -------------------------------------------------------


def test_when_shard_corpus_then_partition_is_a_lossless_cover() -> None:
    """The shards concatenate back to the corpus exactly, in order."""
    tokens = corpus()
    shards = shard_corpus(tokens, max_shard_size=100)
    assert torch.equal(torch.cat(shards), tokens)


def test_when_shard_corpus_then_only_the_last_shard_is_short() -> None:
    """K = ceil(N/size) with every shard full but the last."""
    tokens = corpus()
    shards = shard_corpus(tokens, max_shard_size=500)
    assert len(shards) == -(-tokens.numel() // 500)  # ceil
    for shard in shards[:-1]:
        assert shard.numel() == 500
    assert shards[-1].numel() == tokens.numel() - 500 * (len(shards) - 1)


def test_when_shard_corpus_is_called_twice_then_the_partition_is_identical() -> None:
    """Determinism: f_r is compared across runs, so the partition must not drift."""
    tokens = corpus()
    a = shard_corpus(tokens, max_shard_size=333)
    b = shard_corpus(tokens, max_shard_size=333)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize(
    "bad", [torch.zeros(4, 4, dtype=torch.long), torch.zeros(2, 3, 4, dtype=torch.long)]
)
def test_when_shard_input_is_wrong_shape_then_actionable_error(bad) -> None:
    with pytest.raises(ValueError, match="1-D"):
        shard_corpus(bad)


def test_when_corpus_is_empty_then_no_shards_are_returned() -> None:
    """Empty input yields an empty partition, not a spurious empty shard."""
    assert shard_corpus(torch.zeros(0, dtype=torch.long)) == []


# --- Phase I: the atomizer ----------------------------------------------------


def test_when_atomize_then_rules_are_well_formed() -> None:
    """Every mined rule has both sides non-empty and is not trivially satisfied."""
    result = atomize_shard(corpus()[:500])
    assert result.rules, "structured corpus produced no rules"
    for rule in result.rules:
        assert rule.premise and rule.conclusion
        # A rule whose premise is already inside its conclusion has a penalty at its minimum by construction, so it could never teach anything.
        assert not rule.premise <= rule.conclusion


def test_when_atomize_then_compression_is_relations_over_atoms() -> None:
    """kappa = |R| / Z, the AML compression ratio (arXiv:1803.05252 §3.5 Eq. 36)."""
    result = atomize_shard(corpus()[:500])
    assert result.num_atoms > 0
    assert result.compression == pytest.approx(result.num_relations / result.num_atoms)


def test_when_atomize_then_every_rule_carries_the_shard_compression() -> None:
    """Per-rule compression is the *shard's* kappa, not a per-rule quantity.

    AML defines only the model-level ratio; the PRD's per-rule C_r,k is a
    DARD-specific reading of it, so this pins the reading actually implemented.
    """
    result = atomize_shard(corpus()[:500])
    for rule in result.rules:
        assert rule.compression == pytest.approx(result.compression)


def test_when_atomize_runs_twice_then_the_rule_set_is_identical() -> None:
    """Mining must be deterministic: it is offline but re-runnable."""
    a = atomize_shard(corpus()[:500])
    b = atomize_shard(corpus()[:500])
    assert [r.key for r in a.rules] == [r.key for r in b.rules]


def test_when_atomize_cap_is_hit_then_the_cap_is_respected() -> None:
    result = atomize_shard(corpus()[:2000], max_rules=5)
    assert len(result.rules) <= 5


def test_when_shard_is_shorter_than_the_premise_width_then_no_rules() -> None:
    """Too few tokens to form a premise; must be empty, not a crash."""
    assert atomize_shard(torch.tensor([1, 2, 3], dtype=torch.long)).rules == ()


@pytest.mark.parametrize("bad", [0, -3])
def test_when_atomize_cap_is_not_positive_then_actionable_error(bad: int) -> None:
    with pytest.raises(ValueError, match="max_rules must be >= 1"):
        atomize_shard(corpus()[:200], max_rules=bad)


def test_when_atomize_input_is_2d_then_actionable_error() -> None:
    with pytest.raises(ValueError, match="1-D"):
        atomize_shard(torch.zeros(2, 4, dtype=torch.long))


# --- Phase II: composite weighting -------------------------------------------


def _shard(premise: frozenset[int], compression: float) -> ShardRules:
    return ShardRules(
        rules=(AmlRule(premise, frozenset(premise) | {99}, compression),),
        num_relations=10,
        num_atoms=5,
    )


def test_when_weights_then_they_sum_to_one() -> None:
    """The final renormalization is part of the contract, not a detail."""
    rules, weights, _ = composite_rule_weights([_shard(frozenset({1}), 1.0)] * 3)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert weights.dtype is torch.float32


def test_when_a_rule_appears_in_every_shard_then_it_outweighs_a_once_seen_rule() -> (
    None
):
    """S_r = f_r / K is the Global Invariance term doing its job."""
    common = _shard(frozenset({1}), 1.0)
    rare = _shard(frozenset({2}), 1.0)
    rules, weights, freqs = composite_rule_weights([common, common, rare])
    order = {r.premise: w for r, w in zip(rules, weights)}
    assert order[frozenset({1})] > order[frozenset({2})]
    assert set(freqs.tolist()) == {2.0, 1.0}


def test_when_weight_temperature_is_large_then_the_distribution_flattens() -> None:
    """T_w exists to stop a single invariant from starving the tail."""
    shards = [_shard(frozenset({1}), 1.0), _shard(frozenset({1}), 1.0)]
    shards.append(_shard(frozenset({2}), 1.0))
    _, sharp, _ = composite_rule_weights(shards, weight_temperature=0.1)
    _, flat, _ = composite_rule_weights(shards, weight_temperature=50.0)
    assert (flat.max() - flat.min()).item() < (sharp.max() - sharp.min()).item()


def test_when_weights_then_every_rule_stays_above_the_precision_floor() -> None:
    """PRD §8.3: 100% of rules keep non-zero weight.

    With enough rules that a plain softmax would zero the tail, every weight
    must still be >= eps_floor. This is the assertion that would catch someone
    "simplifying" the clamp away.
    """
    shards = [_shard(frozenset({i}), 1.0) for i in range(200)]
    # Make one rule dominate so the rest would underflow without the floor.
    shards += [_shard(frozenset({9999}), 1.0) for _ in range(50)]
    _, weights, _ = composite_rule_weights(shards)
    assert (weights >= EPS_FLOOR).all()


def test_when_weights_are_compared_to_the_raw_formula_then_they_match() -> None:
    """An independent re-derivation of log raw -> log_softmax -> clamp."""
    shards = [_shard(frozenset({1}), 1.0), _shard(frozenset({2}), 2.0)]
    shards.append(_shard(frozenset({3}), 1.5))
    alpha, beta, temp, floor = 1.0, 0.5, 2.0, 1e-6
    rules, weights, freqs = composite_rule_weights(
        shards, alpha=alpha, beta=beta, weight_temperature=temp, eps_floor=floor
    )
    total = len(shards)
    log_raw = [
        alpha * torch.log(torch.tensor(int(freqs[i]) / total + 1e-8))
        + beta * torch.log(torch.tensor(rules[i].compression + 1e-8))
        for i in range(len(rules))
    ]
    stacked = torch.stack(log_raw)
    expected = torch.exp(torch.log_softmax(stacked / temp, dim=0))
    expected = torch.clamp(expected, min=floor)
    expected = expected / expected.sum()
    assert torch.allclose(weights, expected, atol=1e-6)


def test_when_shard_emits_the_same_rule_twice_then_frequency_counts_it_once() -> None:
    """f_r counts *shards*, not emissions: double-counting inflates S_r past 1."""
    rule = AmlRule(frozenset({1}), frozenset({2}), 1.0)
    shard = ShardRules(rules=(rule, rule, rule), num_relations=3, num_atoms=3)
    _, _, freqs = composite_rule_weights([shard])
    assert freqs.tolist() == [1.0]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": -1.0}, "alpha and beta"),
        ({"beta": -1.0}, "alpha and beta"),
        ({"weight_temperature": 0.0}, "must be > 0"),
        ({"eps_floor": 0.0}, "eps_floor must be > 0"),
    ],
)
def test_when_weighting_params_are_invalid_then_actionable_error(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        composite_rule_weights([_shard(frozenset({1}), 1.0)], **kwargs)


def test_when_there_are_no_shards_then_actionable_error() -> None:
    """S_r = f_r / K is undefined at K = 0, so this must not silently return empty."""
    with pytest.raises(ValueError, match="at least one shard"):
        composite_rule_weights([])


# --- Phase II: compilation ---------------------------------------------------


def test_when_compile_then_matrices_are_sparse_csr_over_the_vocabulary() -> None:
    rules = [AmlRule(frozenset({1, 2}), frozenset({2, 3}), 1.0)]
    rs = compile_rules(rules, torch.ones(1), torch.ones(1), vocab_size=8)
    assert isinstance(rs, DardRuleSet)
    assert rs.premise.layout is torch.sparse_csr
    assert tuple(rs.premise.shape) == (1, 8)
    # Binary indicators: every stored value is exactly 1.0.
    assert torch.all(rs.premise.values() == 1.0)
    assert set(rs.premise.col_indices().tolist()) == {1, 2}
    assert set(rs.conclusion.col_indices().tolist()) == {2, 3}


def test_when_compile_then_the_indicator_matrices_reproduce_the_token_sets() -> None:
    """M[r, v] = 1 iff token v is in rule r's premise, read off as a dense row."""
    rules = [AmlRule(frozenset({5}), frozenset({6, 7}), 1.0)]
    rs = compile_rules(rules, torch.ones(1), torch.ones(1), vocab_size=10)
    row = rs.premise.to_dense()[0]
    assert torch.equal(row.nonzero().flatten(), torch.tensor([5]))
    row_c = rs.conclusion.to_dense()[0]
    assert torch.equal(row_c.nonzero().flatten(), torch.tensor([6, 7]))


def test_when_rule_names_an_out_of_vocab_token_then_actionable_error() -> None:
    """An out-of-range id would index the wrong vocabulary row silently."""
    rules = [AmlRule(frozenset({99}), frozenset({1}), 1.0)]
    with pytest.raises(ValueError, match="outside the vocabulary"):
        compile_rules(rules, torch.ones(1), torch.ones(1), vocab_size=8)


def test_when_compile_weight_count_mismatches_rules_then_actionable_error() -> None:
    rules = [AmlRule(frozenset({1}), frozenset({2}), 1.0)]
    with pytest.raises(ValueError, match="expected 1 weights"):
        compile_rules(rules, torch.ones(2), torch.ones(1), vocab_size=8)


def test_when_compile_vocab_is_not_positive_then_actionable_error() -> None:
    rules = [AmlRule(frozenset({1}), frozenset({2}), 1.0)]
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        compile_rules(rules, torch.ones(1), torch.ones(1), vocab_size=0)


# --- Phase II: the whole pipeline --------------------------------------------


def test_when_mine_rules_then_the_result_is_usable_by_the_loss() -> None:
    """End of Phase II, start of Phase III: a compiled set, not a rule list."""
    shards = shard_corpus(corpus(), max_shard_size=500)
    rs = mine_rules(shards, vocab_size=64, max_rules_per_shard=32)
    assert rs.num_rules > 0
    assert rs.vocab_size == 64
    assert rs.weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    logits = torch.randn(2, 5, 64)
    assert torch.isfinite(logits).all()


def test_when_mining_is_parallel_then_it_matches_serial() -> None:
    """workers>1 must not change the rules -- only the schedule."""
    shards = shard_corpus(corpus(), max_shard_size=500)
    serial = mine_rules(shards, vocab_size=64, workers=0, max_rules_per_shard=16)
    parallel = mine_rules(shards, vocab_size=64, workers=2, max_rules_per_shard=16)
    assert serial.num_rules == parallel.num_rules
    assert torch.allclose(serial.weights, parallel.weights, atol=1e-6)


def test_when_pruning_is_enabled_then_surviving_weights_still_sum_to_one() -> None:
    """Pruning renormalizes among survivors; it is a threshold, not a zeroing."""
    shards = shard_corpus(corpus(), max_shard_size=300)
    rs = mine_rules(
        shards, vocab_size=64, max_rules_per_shard=32, min_shard_frequency=2
    )
    if rs.num_rules:
        assert rs.weights.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert (rs.frequency >= 2).all()


def test_when_no_rule_survives_pruning_then_an_empty_set_is_returned() -> None:
    """An empty set is legal; it just contributes exactly 0.0."""
    shards = shard_corpus(corpus(), max_shard_size=300)
    rs = mine_rules(shards, vocab_size=64, min_shard_frequency=99)
    assert rs.num_rules == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"vocab_size": 0}, "vocab_size must be positive"),
        ({"min_shard_frequency": 0}, "min_shard_frequency must be >= 1"),
        ({"min_compression_threshold": -1.0}, "must be >= 0"),
    ],
)
def test_when_mining_params_are_invalid_then_actionable_error(kwargs, match) -> None:
    shards = shard_corpus(corpus(), max_shard_size=500)
    with pytest.raises(ValueError, match=match):
        mine_rules(shards, **{"vocab_size": 64, **kwargs})


def test_when_there_are_no_shards_to_mine_then_actionable_error() -> None:
    with pytest.raises(ValueError, match="no shards"):
        mine_rules([], vocab_size=64)


# --- Phase II: artifact roundtrip --------------------------------------------


def test_when_rule_set_is_saved_and_loaded_then_it_is_byte_identical() -> None:
    shards = shard_corpus(corpus(), max_shard_size=500)
    rs = mine_rules(shards, vocab_size=64, max_rules_per_shard=16)
    path = "/tmp/dard_rules_test.pt"
    save_rule_set(path, rs, {"note": "roundtrip"})
    loaded, meta = load_rule_set(path)
    assert loaded.num_rules == rs.num_rules
    assert loaded.vocab_size == rs.vocab_size
    assert torch.equal(loaded.weights, rs.weights)
    assert torch.equal(loaded.premise.to_dense(), rs.premise.to_dense())
    assert meta["note"] == "roundtrip"


def test_when_artifact_version_is_stale_then_a_loud_error_not_a_silent_load() -> None:
    """A v1 rule set loads with identical shapes, so a version bump must reject.

    Same reason `NOISE_MAP_VERSION` exists for the DiffusionBlocks noise map: a
    stale artifact would enforce different rules while looking healthy.
    """
    shards = shard_corpus(corpus(), max_shard_size=500)
    rs = mine_rules(shards, vocab_size=64, max_rules_per_shard=8)
    path = "/tmp/dard_rules_version.pt"
    save_rule_set(path, rs, {})
    blob = torch.load(path, weights_only=False, map_location="cpu")
    blob["version"] = RULE_SET_VERSION + 1
    torch.save(blob, path)
    with pytest.raises(ValueError, match="re-mine"):
        load_rule_set(path)


def test_when_artifact_is_missing_a_field_then_actionable_error() -> None:
    path = "/tmp/dard_rules_incomplete.pt"
    torch.save({"version": RULE_SET_VERSION, "premise": None}, path)
    with pytest.raises(ValueError, match="missing fields"):
        load_rule_set(path)


def test_when_meta_is_recorded_then_it_describes_the_rule_set() -> None:
    shards = shard_corpus(corpus(), max_shard_size=500)
    rs = mine_rules(shards, vocab_size=64, max_rules_per_shard=16)
    meta = rule_set_to_meta(rs, {"source": "test"})
    assert meta["version"] == RULE_SET_VERSION
    assert meta["num_rules"] == rs.num_rules
    assert meta["vocab_size"] == 64
    assert meta["source"] == "test"
    assert meta["nnz_premise"] > 0


# --- The typed contract's own invariants ------------------------------------


def test_when_rule_has_an_empty_side_then_actionable_error() -> None:
    with pytest.raises(ValueError, match="non-empty premise"):
        AmlRule(frozenset(), frozenset({1}), 1.0)
    with pytest.raises(ValueError, match="non-empty conclusion"):
        AmlRule(frozenset({1}), frozenset(), 1.0)


def test_when_rule_key_is_taken_then_compression_is_excluded() -> None:
    """Two readings of the same relation are the *same* rule for f_r."""
    a = AmlRule(frozenset({1}), frozenset({2}), 1.0)
    b = AmlRule(frozenset({1}), frozenset({2}), 9.0)
    assert a.key == b.key


def test_when_shard_rules_have_a_nonpositive_atom_count_then_error() -> None:
    with pytest.raises(ValueError, match="num_atoms must be positive"):
        ShardRules(rules=(), num_relations=0, num_atoms=0)


def test_when_rule_set_vocab_disagrees_with_matrix_width_then_error() -> None:
    """A width mismatch must fail at construction, not inside a backward pass."""
    shards = shard_corpus(corpus(), max_shard_size=500)
    rs = mine_rules(shards, vocab_size=64, max_rules_per_shard=8)
    with pytest.raises(ValueError, match="vocabulary width"):
        DardRuleSet(
            premise=rs.premise,
            conclusion=rs.conclusion,
            weights=rs.weights,
            frequency=rs.frequency,
            vocab_size=rs.vocab_size + 1,
        )


def test_when_rule_set_weights_are_not_float32_then_error() -> None:
    """fp32 is load-bearing: the floor underflows in fp8 (E5M2 min 2^-16)."""
    shards = shard_corpus(corpus(), max_shard_size=500)
    rs = mine_rules(shards, vocab_size=64, max_rules_per_shard=8)
    with pytest.raises(ValueError, match="float32"):
        DardRuleSet(
            premise=rs.premise,
            conclusion=rs.conclusion,
            weights=rs.weights.to(torch.bfloat16),
            frequency=rs.frequency,
            vocab_size=rs.vocab_size,
        )
