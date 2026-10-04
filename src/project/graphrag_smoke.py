"""Index and summarize exactly one FINDSum segment using Microsoft GraphRAG."""

import asyncio
from contextlib import chdir, closing
import json
import logging
import os
from pathlib import Path
import time
import traceback
from uuid import uuid4

from .findsum import DEFAULT_DATA_DIR, load_findsum
from .ollama_checks import CHAT_MODEL, EMBEDDING_MODEL, ENDPOINT, check_chat, check_embeddings, check_models

# Use LiteLLM's bundled model metadata rather than fetching it from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY = (
    "Summarize the company's Results of Operations from the supplied document. "
    "Describe revenue, expenses, profitability, year-over-year changes, and their "
    "drivers where available. Preserve reported amounts, units, and periods. "
    "Use only evidence in the document; do not invent missing figures."
)
SMALL_DOCUMENT = (
    "Example Corp owns Retail Division. Retail Division sells consumer goods. "
    "Example Corp reported revenue of EUR 500 million in 2024, up from EUR 450 "
    "million in 2023. Retail Division contributed EUR 300 million in 2024, up "
    "from EUR 260 million in 2023 due to higher sales volumes. Example Corp's "
    "operating income rose from EUR 60 million to EUR 70 million. Higher staff "
    "costs partially offset revenue growth."
)


async def _generate(workspace: Path) -> str:
    import pandas as pd
    from graphrag import api
    from graphrag.config.load_config import load_config
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

    class Progress(NoopWorkflowCallbacks):
        def workflow_start(self, name, instance):
            print(f"Starting: {name}", flush=True)

        def workflow_end(self, name, instance):
            print(f"Finished: {name}", flush=True)

    # GraphRAG 3.2 resolves relative storage and prompt paths against cwd.
    with chdir(workspace):
        config = load_config(workspace)
        print("Indexing exactly one document...", flush=True)
        results = await api.build_index(config=config, callbacks=[Progress()])
        failures = [f"{result.workflow}: {result.error!r}" for result in results if result.error is not None]
        if not results or failures:
            raise RuntimeError(f"GraphRAG indexing failed: {failures}; see {workspace / 'logs'}")

        output = Path(config.output_storage.base_dir)
        documents = pd.read_parquet(output / "documents.parquet")
        if len(documents) != 1:
            raise RuntimeError(f"Expected exactly one indexed document, got {len(documents)}")
        reports = pd.read_parquet(output / "community_reports.parquet")
        entities = pd.read_parquet(output / "entities.parquet")
        relationships = pd.read_parquet(output / "relationships.parquet")
        if entities.empty or relationships.empty:
            raise RuntimeError("GraphRAG did not extract both entities and relationships; inspect logs and cached model responses")
        if reports.empty:
            raise RuntimeError("GraphRAG produced no community reports to summarize")
        print("Generating Results-of-Operations summary...", flush=True)
        response, _ = await api.global_search(
            config=config,
            entities=entities,
            communities=pd.read_parquet(output / "communities.parquet"),
            community_reports=reports,
            community_level=0,
            dynamic_community_selection=False,
            response_type="Three concise paragraphs",
            query=QUERY,
        )
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("GraphRAG returned an empty or non-text summary")
        return response


def local_settings():
    """Read and validate the version-specific configuration before any model calls."""
    import yaml
    from importlib.metadata import version
    from graphrag.config.models.graph_rag_config import GraphRagConfig

    if version("graphrag") != "3.2.0":
        raise RuntimeError("This runner requires the pinned GraphRAG 3.2.0; run poetry install")
    settings = yaml.safe_load((PROJECT_ROOT / "settings.yaml").read_text(encoding="utf-8"))
    for role, provider, model in (
        ("completion_models", "ollama_chat", CHAT_MODEL),
        ("embedding_models", "ollama", EMBEDDING_MODEL),
    ):
        if not settings.get(role):
            raise ValueError(f"Missing local {role} configuration")
        for config in settings[role].values():
            if config["model_provider"] != provider or config["model"] != model or config["api_base"] != ENDPOINT:
                raise ValueError(f"{role} must use the specified local Ollama model and endpoint")
    GraphRagConfig(**settings)
    return settings


def run_smoke_test(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = PROJECT_ROOT / "outputs",
    prepare_only: bool = False,
    small: bool = False,
) -> Path:
    """Use the first loader record; return its unique summary output directory.

    Each invocation gets a fresh index so other documents cannot enter the run.
    The reference summary is saved separately and never supplied to GraphRAG.
    Set prepare_only to stage input/config/reference without calling the API.
    """
    from graphrag.cli.initialize import initialize_project_at
    import yaml

    settings = local_settings()
    if small:
        record = {"report_id": "ollama-small-smoke", "split": "synthetic", "document": SMALL_DOCUMENT, "summary": ""}
    else:
        with closing(load_findsum(data_dir)) as records:
            record = next(records, None)
    if record is None or not record["document"].strip():
        raise ValueError("FINDSum contains no non-empty first document")

    run_id = ("ollama-small-" if small else "findsum-roo-ollama-") + uuid4().hex[:12]
    output_dir = Path(output_dir).resolve()
    workspace = output_dir / "experiments" / run_id
    summaries = output_dir / "summaries" / run_id
    initialize_project_at(workspace, force=False, model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL)
    settings_path = workspace / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    (workspace / ".env").write_text("# Local Ollama: no real API key required.\n", encoding="utf-8")
    (workspace / "input" / "document.txt").write_text(record["document"], encoding="utf-8")
    summaries.mkdir(parents=True)
    (summaries / "reference.txt").write_text(record["summary"], encoding="utf-8")
    (workspace / "reference.txt").write_text(record["summary"], encoding="utf-8")
    metadata = {
        "report_id": record["report_id"],
        "split": record["split"],
        "document_count": 1,
        "query": QUERY,
        "model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "endpoint": ENDPOINT,
        "graphrag_version": "3.2.0",
        "workspace": str(workspace),
        "status": "prepared",
        "summaries": str(summaries),
    }

    def save_metadata() -> None:
        (summaries / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_metadata()
    print(f"Selected {record['report_id']} ({record['split']})", flush=True)
    print(f"Summary directory: {summaries}", flush=True)
    if prepare_only:
        return summaries
    start = time.perf_counter()
    handler = logging.FileHandler(workspace / "warnings-errors.log", encoding="utf-8")
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)
    try:
        metadata["status"] = "checking_models"
        save_metadata()
        print("Checking both local models before indexing...", flush=True)
        metadata["installed_models"] = check_models()
        metadata["embedding_check"] = check_embeddings()
        metadata["chat_check"] = check_chat()
        if metadata["embedding_check"]["dimensions"] != settings["vector_store"]["vector_size"]:
            raise RuntimeError("Embedding dimension does not match configured vector size")
        metadata["status"] = "indexing_and_querying"
        save_metadata()
        response = asyncio.run(_generate(workspace))
        (summaries / "generated.txt").write_text(response, encoding="utf-8")
        (workspace / "generated.txt").write_text(response, encoding="utf-8")
        metadata["status"] = "complete"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        (workspace / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        metadata["runtime_seconds"] = round(time.perf_counter() - start, 3)
        save_metadata()
        logging.getLogger().removeHandler(handler)
        handler.close()
    return summaries
