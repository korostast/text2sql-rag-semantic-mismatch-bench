# Artifacts

This directory contains the outputs and results of the benchmark runs, evaluations and experimental scripts that were written for this research.

## Structure

- `benchmark_runs/`: Contains the raw output files from the LLM benchmark runs.
    - `dev/`: Benchmark runs for the dev dataset (with evidence included in prompts).
    - `mini_dev/`: Benchmark runs for the original mini_dev dataset (with evidence included in prompts).
    - `mini_dev_ru_mod/`: Benchmark runs for the modified Russian mini_dev dataset without evidence.
    - `no_evidence/`: mini_dev benchmark runned without evidence in prompts.
- `evaluation_results/`: Contains the evaluated results of the benchmark runs, including scores and detailed analysis.
    - `dev/`: Evaluation results for the dev dataset.
    - `mini_dev/`: Evaluation results for the original mini_dev dataset.
    - `mini_dev_ru_mod/`: Evaluation results for the modified Russian mini_dev dataset without evidence.
    - `no_evidence/`: Evaluation results for the no-evidence runs.
- `summary/`: Research results plots:
    - Metrics comparison for each experiment (mini_dev, dev, no_evidence, mini_dev_ru_mod) and each model
    - Detailed comparision performance downgrade between baseline and full method when moving from mini_dev to mini_dev_ru_mod
    - Average performance gain by each method
- `error_analysis.png`: Error analysis conducted on mini_dev with evidence and based on `gpt-oss-120b_mini_dev_annotated.json`.
- `gpt-oss-120b_mini_dev_annotated.json`: Manually annotated EX errors of baseline gpt-oss-120b.
