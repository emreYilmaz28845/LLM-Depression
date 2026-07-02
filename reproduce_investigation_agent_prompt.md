# Reproducibility Investigation Agent Prompt

You are a senior ML reproducibility investigator. Your task is to determine why my implementation cannot match the reported results in this paper:

Paper:
`/home/emre/Projects/AudioLLM/Papers/DepresInstruct.pdf`

Repository:
`/home/emre/Projects/AudioLLM/LLM-Depression`

Current result table:
`/home/emre/Projects/AudioLLM/LLM-Depression/depression_results_table_no_emo.csv`

Please do a rigorous investigation, not a surface-level comparison.

## Goals

1. Read the paper carefully and extract the exact experimental setup:
   - datasets used
   - train/validation/test split protocol
   - label definitions and thresholds
   - sample counts and class balances
   - input modalities
   - preprocessing
   - model/backbone details
   - prompt/instruction format
   - optimizer, learning rate, batch size, epochs
   - cross-validation or held-out test details
   - metric definitions, especially F1 type and positive class
   - whether results are best fold, mean fold, test set, or selected checkpoint

2. Inspect the repository implementation:
   - configs under `configs/`
   - dataset loaders under `src/data/`
   - training/evaluation code under `src/train.py`, `src/evaluate.py`, `src/metrics.py`
   - model code under `src/model/`
   - scripts used to run DAIC, EDAIC, CMDC, EATD, and Turkish experiments
   - result aggregation code

3. Compare paper vs implementation line by line.
   Look specifically for mismatches in:
   - dataset splits
   - subject-level vs utterance-level splitting
   - leakage prevention
   - label thresholds
   - positive class convention
   - macro F1 vs positive-class F1
   - checkpoint selection metric
   - validation/test selection
   - prompt templates
   - text/audio preprocessing
   - transcript source
   - emotion features omitted or included
   - frozen vs trainable encoders
   - LoRA/adapters/projectors
   - class imbalance handling
   - seed/fold averaging
   - whether paper results may be single best run rather than mean

4. Use `depression_results_table_no_emo.csv` as the observed evidence.
   Pay attention to the fact that:
   - DAIC text-only is better than the paper, but audio+text and audio-only are worse.
   - CMDC results are close but below paper, especially audio+text.
   - Turkish results have high positive recall but very weak negative F1, suggesting majority-positive bias.
   Explain what each discrepancy implies.

5. Produce a ranked list of likely causes.
   For each cause, include:
   - evidence from the paper
   - evidence from the code/results
   - why it could explain the gap
   - how to verify it
   - exact files/configs/scripts to inspect or modify

6. Do not make code changes initially.
   First produce an investigation report with:
   - Confirmed mismatches
   - Likely mismatches
   - Unclear / needs experiment
   - Most important next experiments
   - Minimal changes to test first

7. After the report, propose a concrete reproduction plan:
   - exact commands to run
   - configs to use or create
   - metrics to log
   - sanity checks
   - expected outcomes that would confirm or reject each hypothesis

Be skeptical. Assume the paper's reported setup may contain missing details, and assume my implementation may differ in subtle but important ways. Prefer evidence from files and tables over speculation.

## Optional Continuation

After completing the investigation report, implement only the smallest experimental changes needed to test the top 2-3 hypotheses, then run or prepare the exact commands needed to compare against the current table.
