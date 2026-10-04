# thesis_project

Phase 1: load the local FINDSum ROO CSV release with no external dependencies.
From this directory, run Python with `PYTHONPATH=src`:

```python
from project.findsum import load_findsum

record = next(load_findsum())
print(record["report_id"], record["split"])
print(record["document"])
print(record["summary"])
```

The loader streams records from `data/text/FINDSum-ROO/roo_input_2000` by
default; pass a directory to override it. Splits are `train`, `validation`,
and `test`. Repeated header rows are skipped. IDs combine the source filename
stem and data row number; they identify segments, not original filings.

Run the small fixture-based tests without loading the dataset:

```sh
PYTHONPATH=src poetry run python -m unittest discover -s tests -v
```

Run the local-only checks and small GraphRAG smoke test from this directory:

```sh
poetry install
poetry run python check_ollama.py chat
poetry run python check_ollama.py embeddings
poetry run python graphrag_smoke.py --small
```

This uses Microsoft GraphRAG **3.2.0**, installed Ollama chat model
**qwen3.8:27b**, and **nomic-embed-text:latest** at `http://localhost:11434`.
`settings.yaml` uses LiteLLM's native `ollama_chat` completion provider and
`ollama` embedding provider. There are no external model calls or real keys.
The literal `ollama-local-dummy` key satisfies GraphRAG 3.2's `ModelConfig`
validation, which requires a nonempty key even for Ollama. `.env` is not needed.

Temperature is zero, thinking is disabled through `reasoning_effort: none`,
requests are serial, retries are limited, and requests have timeouts.
GraphRAG requests JSON when needed for reports and global-search map responses;
entity/relationship extraction uses GraphRAG's own delimiter format. Do not
force JSON globally, because doing so breaks that extraction format.
The verified embedding dimension is 768 and matches the vector store.
The context window is 4096 and completion output is capped at 1024 tokens to
limit memory use. The initial independent chat check took 8 minutes 36 seconds
on this 18 GiB machine with partial CPU offloading, so live GraphRAG runs can
be slow. Freeing memory may help; this runner never substitutes another model.

After the small smoke test succeeds, run one FINDSum example:

```sh
poetry run python graphrag_smoke.py
```

Every run checks both models before indexing. A failed check stops the run.

The first record from the loader is the only document indexed. FINDSum segments
are separate records, so this is one segment, not a reconstructed full filing.
The reference summary is never indexed or included in the query.

Each invocation creates a fresh run with these files:

- `outputs/summaries/<run-id>/generated.txt`: GraphRAG's returned summary.
- `outputs/summaries/<run-id>/reference.txt`: unchanged FINDSum reference.
- `outputs/summaries/<run-id>/metadata.json`: report ID, split, query, models,
  workspace path, and run status.
- `outputs/experiments/<run-id>/`: source document in `input/document.txt`,
  settings, prompts, index, cached model responses, logs, both summaries, and
  metadata including runtime and any error. Runs are named `ollama-small-*`
  or `findsum-roo-ollama-*`.

To prepare the input and reference without an API key or model calls:

```sh
poetry run python graphrag_smoke.py --prepare-only
```

Preparation does not create a generated summary. Failed runs are marked in
metadata and are not reported as successful. Both commands also work with
`.venv/bin/python` instead of `poetry run python` after dependencies are installed.

To run indexing and querying separately, first prepare a workspace:

```sh
poetry run python graphrag_smoke.py --small --prepare-only
# Copy the workspace path from the printed summary directory's metadata.json:
RUN_DIR="/absolute/path/to/outputs/experiments/ollama-small-..."
# Run both check_ollama.py checks successfully before indexing.
LITELLM_LOCAL_MODEL_COST_MAP=True poetry run graphrag index --root "$RUN_DIR"
LITELLM_LOCAL_MODEL_COST_MAP=True poetry run graphrag query --root "$RUN_DIR" --method global --community-level 0 "Summarize the company's Results of Operations."
```

The prepare command performs GraphRAG initialization and copies the local
configuration. Do not rerun bare `graphrag init` over it, since upstream defaults
would replace the local model settings. Use the Python runner to save summaries
and runtime automatically; the direct query command prints its answer.
