"""From-scratch reference implementations used as ablation baselines.

Deliberately a separate package from the sibling ``ablation`` *module*: CPython
resolves a package directory over a same-named module, so keeping
``simple.py`` under ``ablation/`` would silently shadow ``ablation.py`` and
break every ``from nanochat.models.quant.ablation import ...`` site.
"""
