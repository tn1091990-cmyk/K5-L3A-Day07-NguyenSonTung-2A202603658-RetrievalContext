# Day 7 implementation

The implementation is in this package. `corpus.py` reconstructs Day 6 Strategy B chunks with the same paragraph boundaries and IDs; `retrieval.py` supplies Day 6 dense retrieval, BM25, and reciprocal rank fusion; `agent.py` connects an OpenAI-compatible local chat model with search and final answer tools.

See the root `README.md` for configuration, commands, the new benchmark, and what must be reviewed manually.
