NSBM 60-QUESTION BATCH EVALUATION PACK

CONTENTS
- data/evaluation/nsbm_ground_truth.json
  Ground-truth dataset: 60 questions, including 51 answerable and 9 unanswerable.
  The original 21 questions are retained; 39 additional questions are included.
- data/evaluation/nsbm_ground_truth_60.csv
  Spreadsheet-friendly view of all questions, expected answers, evidence files/pages.
- scripts/evaluate_nsbm_batch.py
  Batch retrieval/evidence evaluator, with optional actual answer generation.

INSTALL
Extract this ZIP into the root of your Explainable-Academic-AI repository and allow
it to overwrite data/evaluation/nsbm_ground_truth.json. Keep your existing project
source files and vector database.

RUN RETRIEVAL / EVIDENCE EVALUATION ONLY
  .venv\Scripts\python.exe scripts\evaluate_nsbm_batch.py

RUN THE FULL BATCH INCLUDING GENERATED ANSWERS
  .venv\Scripts\python.exe scripts\evaluate_nsbm_batch.py --generate-answers

The --generate-answers option invokes the configured NSBM RAG agent once per
question (60 calls). It requires the existing .env/model configuration and may
incur API/provider usage charges. Without the flag, the script tests retrieval,
answerability validation, source matching, and supporting-fact coverage.

REPORTS
- data/evaluation/nsbm_batch_evaluation_report.json
- data/evaluation/nsbm_batch_evaluation_report.csv

IMPORTANT EVALUATION LIMITATIONS
- Supporting-fact coverage and generated-answer fact coverage are lexical proxies,
  not semantic entailment or an LLM-judge faithfulness score.
- The abstention marker rate is a phrase-based proxy.
- Review generated answers and citations for high-stakes/current university rules.
- The 2016 BSc/BM/BA by-laws source is explicitly historical; current validity has
  not been verified.
- The 2026 EC web snapshot says 14 days; the older August 2023 PDF says 14 working
  days. Do not conflate these different source versions.
- The 2025 curriculum revision notice does not contain the revised module-by-module
  curriculum. Questions asking those specifics are intentionally unanswerable.
