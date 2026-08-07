---
name: git-pr
description: Package agent-made code, configuration, or methodology changes into a reviewable GitHub pull request using this repository's agent branch convention, experiment-provenance fields, focused commits, and required verification. Use after an agent completes and validates an in-scope change, or when the user asks to commit, push, or open a PR. Open the PR for the user to review and merge; never merge it.
---

# Publish an agent change for review

Treat Git branches and PRs as part of experiment provenance. Read the GitHub workflow in `docs/AudioLLM_Experiment_Workflow_Implementation_Plan_v2.md` before publishing experiment or methodology changes. Use the repository `AGENTS.md` rule for agent-authored branches: `agent/<topic>` targeting `main`. Preserve the plan's semantic prefix and Issue number inside the topic when useful, for example `agent/exp-86-daic-rotary-k` or `agent/fix-91-eval-view`.

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

Agents are pre-authorized by `AGENTS.md` to commit their own in-scope changes, push `agent/<topic>`, and open a PR to `main`. This authorization does not cover merging the PR, force-pushing, unrelated files, or any cluster mutation.

```bash
git push -u origin agent/<topic>
gh pr create --base main --head agent/<topic> --title "<short title>" --body-file /tmp/pr_body.md
```

Open a normal review-ready PR after the acceptance gate passes. Use a draft PR only when the user asks for an early review or a clearly identified required check remains. Never merge the PR; return its URL so the user can review and merge it.

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

Report the branch, commit SHA, verification, PR URL, and anything deliberately excluded from the commit.
