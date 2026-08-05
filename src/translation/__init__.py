"""Label-free English transcript translation pipeline for non-English datasets.

Stages: translation-unit export (``src.translation.units``), deterministic
Qwen3.6-27B translation on MN5 (``src.translation.translate``), structural and
semantic validation (``src.translation.validate``), and config-selected
manifest overlay (``src.translation.overlay``). No depression labels, scores,
folds, diagnoses, or subject-class metadata ever reach the translator.
"""
