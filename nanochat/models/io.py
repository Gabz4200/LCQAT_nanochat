"""Typed contracts for values that cross a module boundary.

A positional tuple cannot express a precondition between its own fields, so
every unpack site has to re-derive it from the surrounding context. The clearest
case is `LayerQuantSpec`: `out_spec` is only meaningful when `quantize_out` is
True, and the two used to travel as a 4-tuple, so every unpack site carried a
bare `out_spec` with nothing recording that it was inert otherwise.

Named fields carry the same information with the invariant stated once, in the
type, instead of in a docstring each unpack site has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A codebook cardinality: a total level count `K`, or an explicit
#: ``(m_neg, m_pos)`` split around the zero anchor. Defined here rather than in
#: `lcqat.retrofit` so that both the contract and its consumer can import it
#: without a cycle.
CodebookSpec = int | tuple[int, int]


@dataclass(frozen=True)
class LayerQuantSpec:
    """The quantization plan for one layer.

    Attributes:
        weight: codebook spec for the layer's weight matrix. The weight
            quantizer is never sigma-conditioned -- one weight matrix feeds
            every noise level -- so this is always an unconditional spec.
        activation: codebook spec for the layer's input activations. May be
            sigma-conditioned when a conditioned codebook is installed.
        quantize_output: whether the layer's output is quantized at all. When
            False the layer has no `out_quantizer`.
        output: output-quantizer spec. Only meaningful when
            `quantize_output` is True; `__post_init__` normalizes the
            meaningless case to the activation spec so the value is never
            stale, rather than leaving a caller to pass `None` and unpack it
            unconditionally.
    """

    weight: CodebookSpec
    activation: CodebookSpec
    quantize_output: bool = False
    output: CodebookSpec | None = None

    def __post_init__(self) -> None:
        if self.output is None:
            # Inert when the output is not quantized; mirroring the
            # activation spec is what `LCQATLinear` does when `out_k is None`.
            object.__setattr__(self, "output", self.activation)

    @property
    def effective_output(self) -> CodebookSpec:
        """The output spec to build with, valid whether or not it is used."""
        return self.output if self.output is not None else self.activation

    def as_tuple(self) -> tuple[CodebookSpec, CodebookSpec, bool, CodebookSpec]:
        """Backwards-compatible positional form, for callers not yet migrated.

        Kept so the migration is mechanical rather than a flag day. Prefer
        attribute access at new call sites; delete this once no caller uses it.
        """
        return (
            self.weight,
            self.activation,
            self.quantize_output,
            self.effective_output,
        )
