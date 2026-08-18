"""Qwen3.8-27B MN5 offline deployment and Turkish question-recovery package.

Submodules:

- ``contracts``: fixed pins, enums, schemas, filename parsing, and the
  section-17 serving-configuration selection rules.
- ``validation``: synthetic-case validation harness and acceptance gating.
- ``turkish_questions``: private Turkish subject-sequence preparation,
  resumable inference, two-level consolidation, and deterministic rendering.
- ``audit``: wheel-tag, deployment, and Turkish compact-evidence audits.

All code in this package is stdlib-only. It never imports torch, vLLM,
transformers, or the OpenAI SDK; those are used only by the CLI scripts in
``scripts/``.
"""
