---
name: git-pr
description: Package agent-made code, configuration, or methodology changes into a reviewable GitHub pull request using this repository's agent branch convention, experiment-provenance fields, focused commits, required verification, and controlled merging when a recorded task-specific full-autonomy grant permits it. Use after an agent completes and validates an in-scope change, when the user asks to commit, push, open, or merge a PR, or when an autonomous task must continue across PR boundaries.
---

# Publish and, when authorized, merge an agent change

Treat Git branches and PRs as part of experiment provenance. Read the GitHub workflow in `docs/AudioLLM_Experiment_Workflow_Implementation_Plan_v2.md` before publishing experiment or methodology changes. Use the repository `AGENTS.md` rule for agent-authored branches: `agent/<topic>` targeting `main`. Preserve the plan's semantic prefix and Issue number inside the topic when useful, for example `agent/exp-86-daic-rotary-k` or `agent/fix-91-eval-view`.

## Resolve merge authority

- Standing repository authorization covers committing the agent's own in-scope changes, pushing `agent/<topic>`, and opening or updating PRs. It does not cover merging.
- A task-specific full-autonomy grant covers merging the agent's own PRs for that named task after the grant has been journaled as required by `AGENTS.md`.
- Without an active recorded full-autonomy grant, never merge. Return the PR URL for user review.
- A runbook, Issue, silence, prior-session grant, or general repository instruction is not a task-specific grant.
- Full autonomy does not authorize merging unrelated PRs, bypassing branch protection, using an administrative or forced merge, ignoring requested changes, or expanding the task.

## Preserve the experiment hierarchy

- Put one coherent code, bug-fix, configuration, or methodology change on one branch.
- Do not create separate branches for seeds or folds. One PR may produce many logical runs, attempts, seeds, and folds.
- Link the Issue when one defines the research question or acceptance criterion.
- Record the full commit SHA, branch, clean/dirty state, and Issue/PR identifiers in experiment metadata or deployment provenance.
- Base merge recommendations on correctness, maintainability, and scientific validity, not whether a metric improved.

## Prepare the branch safely

Inspect before changing Git state:

```bash
git status --short
git branch --show-current
git log --oneline -10
git diff --stat
git diff
git fetch origin
```

- Preserve unrelated user changes and stage only files changed for the task.
- If already on the correct `agent/<topic>` branch, continue there.
- If the task changes are uncommitted on `main`, create `agent/<topic>` at the current commit so the changes remain in place. Do not switch to and pull `main` first.
- If starting from a clean tree, create the branch from the reviewed `origin/main` state.
- Do not reset, force-push, rewrite published history, or rebase across unrelated work unless the user explicitly requests it.

Examples:

```bash
# Preserve in-progress changes already based on the current commit.
git switch -c agent/<topic>

# Start a new clean change from the reviewed remote main branch.
git switch -c agent/<topic> origin/main
```

## Commit the intended change

- Review `git diff` and `git diff --cached` before committing.
- Never stage secrets, credentials, `.env`, `~/.netrc`, or unrelated files.
- Respect ignored runtime evidence. Force-add only an exact requested file under an ignored path, never an ignored directory.
- Use a short imperative commit message with no trailing period.
- Run the acceptance tests appropriate to the change in `llmdep4090`; use `python -m pytest`, never bare `pytest`.

## Push and open the PR

Agents are pre-authorized by `AGENTS.md` to commit their own in-scope changes, push `agent/<topic>`, and open a PR to `main`. Without a recorded full-autonomy grant, this authorization does not cover merging the PR. It never covers force-pushing, unrelated files, or cluster mutation.

```bash
git push -u origin agent/<topic>
gh pr create --base main --head agent/<topic> --title "<short title>" --body-file /tmp/pr_body.md
```

Open a normal review-ready PR after the acceptance gate passes. Use a draft PR only when the user asks for an early review or a clearly identified required check remains. If no active recorded full-autonomy grant permits merging, return the PR URL so the user can review and merge it.

Use this body structure:

```markdown
## Task / scientific change
<coherent change, research question, and Issue/task-card link when applicable>

## Files created
- ...

## Files modified
- ...

## Commands run / verification
- exact command and pass/fail result

## Artifacts produced
- paths, or none

## Experiment and provenance impact
- branch and full commit SHA
- affected configs/protocols
- related group, logical-run, attempt, Issue, and PR identifiers when available
- generated group-report summary when results exist; every number must link to complete provenance

## Known limitations
- ... or none

## Handoff
- unrelated pre-existing worktree changes preserved
- next task, explicitly not started
```

Keep the title concise and topic-scoped. If the branch already has an open PR, push the new commit and update that PR instead of creating a duplicate.

## Merge under full autonomy

When an active recorded full-autonomy grant covers the named task:

1. Confirm the PR contains only the agent's in-scope task changes and targets the intended base branch.
2. Confirm the PR head SHA is the exact locally validated commit and the worktree contains no uncommitted task changes.
3. Inspect required checks and review state. Do not merge while required checks fail or are pending, the PR is draft, or a review requests changes.
4. Resolve in-scope failures or review comments, rerun the required local validation, push the fix, and recheck the PR. Stop if resolution requires scope expansion or an excluded action.
5. Merge with the repository's established merge-commit method. Do not use admin bypass, a forced merge, force-push, or history rewriting. Preserve the source branch unless deletion is separately authorized.
6. Verify the resulting merge commit on the remote base branch and record the PR, head SHA, and merge SHA in the journal and any experiment provenance.
7. For prerequisite or stacked PRs, merge in dependency order. Rebase or retarget the next PR only when it can be done without rewriting published history; otherwise merge the base, create a clean follow-up branch from the updated base, reapply only the intended change, and rerun validation.
8. Continue the named task without waiting for user review unless the grant expired or a hard-stop condition was reached.

Use the normal GitHub merge path, matching this repository's merge-commit history. A typical command is:

```bash
gh pr merge <pr-number> --merge
```

Do not pass an admin flag or request branch deletion.

Report the branch, validated head SHA, verification, PR URL, merge status and merge SHA when applicable, and anything deliberately excluded from the commit.

## Parallel workflow — stacked branches and PR non-blocking

- Allow dependent PRs to target recorded parent branches (e.g., `agent/exp-base-pooling` targeting `agent/exp-base`), not only `main`; record parent branch/SHA and dependency PR in `.agent-pin.json` and `experiments/definitions/`
- Tier 1 (`agent/exp-*`) winner integrations normally squash; Tier 2 (`agent/feat-*`) and integration PRs may use merge commits when history helps review/bisect
- A PR is not an experiment pause: a lane opens a draft/normal PR for review/provenance but continues to deploy and iterate from its exact branch commit; a dependent lane branches from the required unmerged commit and records the dependency
- Preserve no-force-push, checks, review, and grant-based merge safeguards; ambiguous cross-lane integration is assigned to the orchestrator/human decision boundary and must not be auto-merged

