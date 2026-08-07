---
name: agent-journal
description: Write and maintain this repository's agent journal after meaningful work, including PR work, experiments and training runs, experiment results, debugging findings, architecture or methodology decisions, reproducibility changes, and blockers. Use after a meaningful milestone is known; not for trivial edits, routine inspection, or status checks with no new finding.
---

# Maintain the agent journal

The journal is the narrative index of meaningful agent work in this repo. Entries live in `docs/agent-journal/`, one file per day, written in plain English and appended, never rewritten.

## When to write

Write an entry after any of these milestones:

- Meaningful PR work or merged change sets
- Experiments or training runs and their results
- Important debugging findings
- Architecture or design decisions
- Methodology changes
- Reproducibility changes
- A significant blocker

Do not journal trivial edits, routine inspection, or status checks that produced no new finding. Write the entry after the milestone result is known, not while it is still in flight.

## Where to write

- One file per day: `docs/agent-journal/YYYY-MM-DD.md`, using the `Europe/Istanbul` calendar date.
- Append multiple entries to the same daily file; never overwrite earlier entries.
- If a PR or experiment later gets a new ID or outcome, append a new entry instead of rewriting history.

## What to include

- The context and why the work was needed.
- The decision and its reason, or what changed or ran.
- The hypothesis or expected outcome when relevant.
- The actual result or current state.
- What should happen next.
- Real references when available: experiment/run/attempt/job IDs, branch, full commit SHA, PR, checkpoint, config, dataset version, and evidence paths. Never invent identifiers.

## What to avoid

- Secrets, credentials, raw transcripts, subject identifiers, or sensitive dataset content.
- Results without provenance — apply the repo's reporting rule to every result written in the journal.
- Rewriting history: the journal is a narrative index, not experiment evidence. Keep `run_config.yaml`, tracking sidecars, local artifacts, generated reports, and PRs authoritative.

## Style

Write entries in plain English (see the `plain-english` skill).
