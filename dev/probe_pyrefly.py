from typing import reveal_type

from nanochat.models.quant.codebook import MemoryEfficientLearnedCodebook
from nanochat.models.quant.learnable_lut import LearnableIndexLut

c_in = MemoryEfficientLearnedCodebook(m_neg=7, m_pos=7).get_codebook()
c_out = MemoryEfficientLearnedCodebook(m_neg=7, m_pos=7).get_codebook()
lut = LearnableIndexLut(c_in, c_out, "relu2", relaxation="proximity")
reveal_type(lut.pin_mask)
reveal_type(lut.knots)
reveal_type(lut.levels)
