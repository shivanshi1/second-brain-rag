# Evaluation results

Run the golden set against the local API after ingesting `eval/corpus`:

```powershell
python -m secondbrain.evaluation.runner --repeats 3 --top-k 5 --output eval/results/baseline.json
```

The evaluator calls `/search` and strict `/ask` for each question on every repeat. It reports
source-level Recall@k (fraction of expected source titles found), MRR, binary nDCG,
citation-marker validity, the API's NLI-supported sentence ratio, refusal rates, exact answer
consistency across repeats, and ask latency. It does not decide whether prose is factually correct:
manually review answers and citations, especially where the NLI score is high.

## Golden set

`eval/golden.json` is a starter set for the two sample documents, not a statistically meaningful
benchmark. Add questions and independently verify expected sources before comparing versions.
The unanswerable cases have no expected source and check strict refusal behavior.

## Results

No benchmark numbers have been recorded yet. Run the command above on the same corpus, model,
and settings before and after retrieval changes. Keep the generated JSON report with the exact
configuration used; do not fill in estimated or invented scores.

| Experiment | Recall@5 | MRR | nDCG | Citation-valid rate | NLI support ratio | Correct refusal rate | Exact answer consistency | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hybrid RRF baseline | pending | pending | pending | pending | pending | pending | pending | Current retrieval; no reranker or relevance gate. |
