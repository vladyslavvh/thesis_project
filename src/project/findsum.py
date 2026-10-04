"""Load FINDSum ROO segments without modifying their text."""

import csv
from pathlib import Path
from typing import Iterator, TypedDict


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data/text/FINDSum-ROO/roo_input_2000"
)


class FINDSumRecord(TypedDict):
    report_id: str
    document: str
    summary: str
    split: str


def load_findsum(data_dir: str | Path = DEFAULT_DATA_DIR) -> Iterator[FINDSumRecord]:
    """Yield ROO rows in filename order, using UTF-8 CSV decoding.

    IDs are synthetic source-row identifiers (filename stem plus 1-based data
    row number), not original filing IDs. Segments remain separate records.
    ``val`` filenames map to ``validation``. Only obvious repeated headers
    are removed; document and summary text are preserved verbatim.

    Validation and I/O happen when the iterator is consumed.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"FINDSum directory not found: {data_dir}")
    splits = {"train": "train", "val": "validation", "test": "test"}
    paths = sorted(
        path
        for path in data_dir.glob("*_roo_segment_*_input_2_1000.csv")
        if path.name.split("_", 1)[0] in splits
    )
    if not paths:
        raise FileNotFoundError(f"No FINDSum ROO CSV files found in {data_dir}")

    for path in paths:
        split = splits[path.name.split("_", 1)[0]]
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not {"document", "summary"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: expected document and summary columns")
            for row_number, row in enumerate(reader, start=1):
                document, summary = row["document"], row["summary"]
                if document is None or summary is None or None in row:
                    raise ValueError(f"{path}: malformed CSV data row {row_number}")
                if (
                    document.strip().casefold() == "document"
                    and summary.strip().casefold() == "summary"
                ):
                    continue
                yield {
                    "report_id": f"{path.stem}:{row_number}",
                    "document": document,
                    "summary": summary,
                    "split": split,
                }
