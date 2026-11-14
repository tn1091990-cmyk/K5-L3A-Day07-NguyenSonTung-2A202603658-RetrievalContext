# Day 7 — Agentic RAG implementation

**Implementation status:** complete for the standard Day 7 lab. The real agent, three retrieval methods, 24-case benchmark, retrieval comparisons, selected agent dev/test runs, manually reviewed answer/citation/abstention metrics, traces, and report are present. Retrieval uses PostgreSQL/pgvector and the Day 6 embedding model; the agent uses Gemma 4 E4B IT through a separate local chat endpoint. The 15-test mock suite passed.

This solution reuses the 30 HTML snapshots and PostgreSQL/pgvector index from Day 6. It compares three retrieval methods: Day 6 Strategy B dense search, BM25 lexical search, and dense+BM25 reciprocal rank fusion (RRF). The agent uses a real OpenAI-compatible chat endpoint and cites retrieved chunk IDs.

## Day 6 connection

The Day 6 `.env` is loaded at runtime and its values are not copied into this repository or printed. By default, data is read from the sibling Day 6 lab at `C:\Users\DELL\Nextcloud\AIinAction\Day6Lab\K5-L3-Day06-2A202603658-NguyenSonTung-DataIngestionEngineering`. Override with `--day6-root` or `DAY6_ROOT`.

The index contract recorded in the Day 6 report is PostgreSQL 16 + pgvector, Strategy B, `embeddinggemma-2`, 768 dimensions, cosine distance. Runtime dimension discovery is used, and incompatible vector dimensions fail instead of being silently changed. The old Day 6 venv has a stale interpreter path; use a working Python 3.11+ environment and install `starter/requirements.txt` from Day 6.

For embeddings, Day 6 expects its local OpenAI-compatible endpoint (defaults to `http://127.0.0.1:9931/v1`) and a running PostgreSQL instance. The chat endpoint is separate: set `LLM_BASE_URL=http://127.0.0.1:9932/v1` and `LLM_MODEL=gemma-4-E4B-it-UD-Q4_K_XL.gguf`. The embedding endpoint remains on port 9931. `LLM_API_KEY` can override endpoint authentication. Do not put secrets in tracked files.

## Benchmark

`data/benchmark_day7.json` is generated from exact text in the Day 6 paragraph chunks. It contains dev 15 answerable + 2 evidence-absent cases and a fresh test 5 answerable + 2 evidence-absent cases. The fresh test questions use five CPU pages not used as Day 6 gold sources. Gold evidence is not sent to retrieval or the answering model. The two missing-evidence questions on each split ask about warranty, current Vietnam retail price, motherboard USB ports, or display count, none of which is specified in the CPU corpus.

Build the Day 7 corpus and benchmark without contacting the database:

```powershell
py -3 -c "from pathlib import Path; from src.corpus import write_chunks; from src.benchmark import write_benchmark; d6=Path(r'C:\Users\DELL\Nextcloud\AIinAction\Day6Lab\K5-L3-Day06-2A202603658-NguyenSonTung-DataIngestionEngineering'); chunks=write_chunks(d6/'starter'/'data', Path('data/day6_paragraph_chunks.json')); write_benchmark(chunks, Path('data/benchmark_day7.json')); print(len(chunks))"
```

Run the retrieval configurations on dev:

```powershell
py -3 -m src.run_retrieval --split dev --output results/retrieval-dev
```

With the local services offline, the measured lexical-only experiments are available with:

```powershell
py -3 -m src.run_retrieval --split dev --offline-lexical-only --output results/retrieval-dev-offline
py -3 -m src.run_retrieval --split test --offline-lexical-only --output results/retrieval-test-offline
```

Dev configurations: dense Strategy B top-3; BM25 (`k1=1.5`, `b=.75`) top-3; Hybrid RRF-60 top-3; BM25 (`k1=.8`, `b=.25`) top-3; that tuned BM25 with 3,600-character context; and Hybrid RRF-60 with 1,200-character context. The first three are distinct methods; the rest test ranking/context trade-offs. Test runs only the BM25 baseline and configuration selected on dev.

```powershell
py -3 -m src.run_retrieval --split test --output results/retrieval-test
```

Agent run (one selected retrieval method at a time):

```powershell
py -3 -m src.run_agent --split dev --method bm25 --bm25-k1 0.8 --bm25-b 0.25 --top-k 3 --budget-chars 3600 --output results/agent-bm25-dev
py -3 -m src.run_agent --split test --method bm25 --bm25-k1 1.5 --bm25-b 0.75 --top-k 3 --budget-chars 2400 --output results/agent-bm25-baseline-test
py -3 -m src.run_agent --split test --method bm25 --bm25-k1 0.8 --bm25-b 0.25 --top-k 3 --budget-chars 3600 --output results/agent-bm25-selected-test
```

Each agent run saves tool-call traces, returned evidence, final context, answer/citations, errors, and pending human-review labels. The following baseline and selected test runs are already recorded; reproduce them with:

```powershell
py -3 -m src.run_agent --split test --method dense --output results/agent-dense-test
```

Do not report a score for a failed local service/model call. `reviews_pending.json` is a template for future reruns. For the recorded runs, `reviews.json` contains completed human labels and `agent_metrics.json` contains the aggregates. The code checks citation IDs, not semantic support, so labels were assigned after reading each answer and citation against its source chunk. The original Day 6 test results have already been viewed, so they are not treated as Day 7's fresh test data.

## Findings inherited from Day 6

Day 6 reported 30 HTML source units and 270 paragraph chunks for Strategy B. Saved retrieval results show Hit@1 of 13/15 on its dev questions and 3/5 on its old test questions for Strategies B/C; exact Evidence Hit@3 was reported as 0/15 dev and 2/5 old test for B/C. Strategy C did not improve on B. Those old test values are historical Day 6 results, not Day 7 evaluation.

## Day 7 measurements so far

On dev, dense scored Hit@1 13/15, Evidence Hit@3 14/15, Context Evidence Rate 14/15. BM25 baseline scored 11/15, 15/15, 14/15; Hybrid scored 13/15, 15/15, 14/15. Tuned BM25 scored 14/15 Hit@1 and 15/15 for both evidence metrics. Hybrid with budget 1200 retained Evidence Hit@3 15/15 but context evidence fell to 13/15. The dev-selected tuned BM25 with budget 3600 scored 14/15, 15/15, 15/15. On held-out test, the BM25 baseline scored 4/5, 5/5, 5/5; selected BM25 scored 5/5 for all three. Test traces contain only the baseline and selected setup.

The real dense/hybrid comparison and agent audit are complete. Agent metrics after manual review: dev 15/15 fully grounded answerable cases, 21/21 supported citations, 2/2 correct abstentions; selected test 5/5 grounded, 8/8 supported citations, 2/2 correct abstentions. Baseline test: 5/5 grounded, 9/9 supported citations, 2/2 correct abstentions. Gemma sometimes selected the abstain tool despite providing a supported cited answer; raw decisions are preserved and narrowly recovered when citation is in final context and refusal language is absent. Local model latency is approximately 53–73 seconds per case. See the student report and `results/agent-bm25-*/agent_metrics.json`.

## Limitations and reproducibility

The CPU specification collection is a narrow corpus from TechPowerUp snapshots. Day 6 reports that reuse rights were not confirmed. Runtime, dependency versions, model availability, and local power cost should be recorded from the machine running the experiment. BM25 uses Unicode word tokens without Vietnamese word segmentation. There is no fabricated model output or fallback: unavailable services produce an error trace.
