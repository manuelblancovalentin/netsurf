"""Pinned NLP row identities and disjoint NetSuRF fault-exposure cohorts.

Kona owns the assets and their READY handoff. These rules only interpret rows
from files that the asset consumer has already hash-verified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Sst2Row:
    split: str
    source_index: int
    text: str
    label: int


def sst2_records_digest(rows: Sequence[Sst2Row]) -> str:
    """Kona's ordered idx/label/sentence digest, including row order."""
    digest = hashlib.sha256()
    for row in rows:
        sentence = hashlib.sha256(row.text.encode("utf-8")).hexdigest()
        digest.update(f"{row.source_index}\t{row.label}\t{sentence}\n".encode("ascii"))
    return digest.hexdigest()


def read_sst2_parquet(path: Path, split: str) -> tuple[Sst2Row, ...]:
    if split not in {"train", "validation"}:
        raise ValueError("only labeled SST-2 train and validation splits are supported")
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if set(table.column_names) != {"sentence", "label", "idx"}:
        raise ValueError("SST-2 source/label columns differ from the pinned protocol")
    data = table.to_pydict()
    rows = tuple(Sst2Row(split, index, sentence, label)
                 for index, sentence, label in zip(data["idx"], data["sentence"], data["label"]))
    if len(rows) != table.num_rows or len({row.source_index for row in rows}) != len(rows):
        raise ValueError("SST-2 rows have duplicate or missing source identities")
    if any(type(row.source_index) is not int or type(row.text) is not str or
           type(row.label) is not int or row.label not in (0, 1) for row in rows):
        raise ValueError("SST-2 rows have missing text or public class labels")
    return rows


def sst2_fault_partitions(train: Sequence[Sst2Row], validation: Sequence[Sst2Row], *,
                          engineering_count: int = 32, calibration_count: int = 512,
                          selection_count: int = 128) -> dict[str, tuple[Sst2Row, ...]]:
    """Prespecified fault-outcome exposure boundaries within Kona's split rule.

    Calibration uses Kona's first 512 hash-ranked train records; selection uses
    the next 128. The first validation rows are engineering-only. All remaining
    validation rows are reserved for the later frozen final campaign.
    """
    if min(engineering_count, calibration_count, selection_count) < 1:
        raise ValueError("all cohort sizes must be positive")
    if len(validation) <= engineering_count or len(train) < calibration_count + selection_count:
        raise ValueError("insufficient rows for disjoint fault-exposure cohorts")
    if any(row.split != "train" for row in train) or any(row.split != "validation" for row in validation):
        raise ValueError("SST-2 train and validation source splits are required")
    if len({row.source_index for row in train}) != len(train) or len({row.source_index for row in validation}) != len(validation):
        raise ValueError("duplicate SST-2 source identities")
    ranking = sorted(train, key=lambda row: hashlib.sha256(
        f"sst2-train-{row.source_index}".encode("ascii")).hexdigest())
    result = {"calibration": tuple(ranking[:calibration_count]),
              "selection": tuple(ranking[calibration_count:calibration_count + selection_count]),
              "engineering": tuple(validation[:engineering_count]),
              "evaluation": tuple(validation[engineering_count:])}
    identities = [(row.split, row.source_index) for rows in result.values() for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("fault-exposure cohorts overlap")
    return result


def read_wikitext_parquet(path: Path) -> tuple[str, ...]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if set(table.column_names) != {"text"}:
        raise ValueError("WikiText-2 raw text schema changed")
    rows = tuple(table.to_pydict()["text"])
    if not rows or any(type(row) is not str for row in rows):
        raise ValueError("WikiText-2 text rows missing")
    return rows


def wikitext_joined_text(rows: Sequence[str]) -> tuple[str, str]:
    if not rows or any(type(row) is not str for row in rows):
        raise ValueError("nonempty WikiText-2 text rows required")
    text = "\n\n".join(rows)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()
