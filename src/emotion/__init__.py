"""Offline SECap emotion-caption extraction + translation drivers.

These modules import the SECap stack (Chinese HuBERT + Q-Former + Chinese-LLaMA)
and/or translation models and are meant to run ONLY in the isolated ``secap``
conda env, OFFLINE, once. They produce a frozen JSONL cache under
``outputs/emotion/`` that the training/eval code reads by ``sample_id``.

The training code (``src/data/...``, ``src/train.py``, ``src/evaluate.py``) MUST
NOT import anything from this package — that would pull the heavy/non-deterministic
SECap deps into the training env and risk violating the eval-determinism rule.
"""
