"""Agricultural EN<->BN glossary.

Source of truth: AgriBot's dialect knowledge graph (SQLite), which already pairs
canonical Bengali and English agricultural terms (crops, diseases, pests,
chemicals, fertilizers, symptoms) plus dialect aliases. We export it to a flat
glossary used downstream for:

  * mining   - keep parallel pairs whose EN side contains a glossary term and
               whose BN side contains the expected Bengali rendering;
  * constraints - force the canonical Bengali term into the decoded output;
  * TermAcc  - score whether each source term was rendered correctly.

Run directly to (re)generate data/glossary.tsv and data/glossary.json:

    python -m agri_mt.glossary
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

# AgriBot's seeded KG. Override with AGRIBOT_KG_DB if relocated.
import os

DEFAULT_KG_DB = Path(
    os.environ.get("AGRIBOT_KG_DB", r"F:/Projects/Agri_bot/data/knowledge_graph.db")
)
HERE = Path(__file__).resolve().parent.parent
GLOSSARY_TSV = HERE / "data" / "glossary.tsv"
GLOSSARY_JSON = HERE / "data" / "glossary.json"
ADDITIONS_JSON = HERE / "data" / "glossary_additions.json"  # human-curated extras


@dataclass
class GlossaryEntry:
    en: str                              # canonical English term
    bn: str                              # canonical Bengali term
    entity_type: str
    en_aliases: list[str] = field(default_factory=list)
    bn_aliases: list[str] = field(default_factory=list)

    def all_en(self) -> list[str]:
        return _dedup([self.en, *self.en_aliases])

    def all_bn(self) -> list[str]:
        return _dedup([self.bn, *self.bn_aliases])


def _dedup(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        it = (it or "").strip()
        if it:
            seen.setdefault(it, None)
    return list(seen)


def load_glossary(db_path: Path | str = DEFAULT_KG_DB) -> list[GlossaryEntry]:
    """Read entities + aliases from the KG and build glossary entries.

    Aliases carry a ``dialect_region`` of ``english`` for English surface forms;
    everything else (``standard``/``colloquial``/region names) is Bengali.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"KG database not found at {db_path}. Set AGRIBOT_KG_DB to its location."
        )
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    entries: list[GlossaryEntry] = []
    for row in conn.execute(
        "SELECT id, canonical_bn, canonical_en, entity_type FROM entities"
    ).fetchall():
        aliases = conn.execute(
            "SELECT alias_text, dialect_region FROM aliases WHERE entity_id = ?",
            (row["id"],),
        ).fetchall()
        en_aliases, bn_aliases = [], []
        for a in aliases:
            (en_aliases if a["dialect_region"] == "english" else bn_aliases).append(
                a["alias_text"]
            )
        entries.append(
            GlossaryEntry(
                en=row["canonical_en"],
                bn=row["canonical_bn"],
                entity_type=row["entity_type"],
                en_aliases=_dedup(en_aliases),
                bn_aliases=_dedup(bn_aliases),
            )
        )
    conn.close()
    return entries


def merge_additions(entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
    """Fold human-curated extras in, keeping TermAcc honest:

    * if the Bengali form is already known (canonical or alias) -> skip;
    * else if the English term matches an existing entry -> add as a bn alias
      (so one English term is never counted by two entries);
    * else -> a new entry.
    """
    if not ADDITIONS_JSON.exists():
        return entries
    by_en: dict[str, GlossaryEntry] = {e.en.lower(): e for e in entries}
    for e in entries:
        for a in e.en_aliases:
            by_en.setdefault(a.lower(), e)
    known_bn: set[str] = set()
    for e in entries:
        known_bn.update(e.all_bn())

    for add in json.loads(ADDITIONS_JSON.read_text(encoding="utf-8")):
        en, bn, typ = add["en"].strip(), add["bn"].strip(), add.get("type", "term")
        if not en or not bn or bn in known_bn:
            continue
        host = by_en.get(en.lower())
        if host is not None:
            if bn not in host.bn_aliases and bn != host.bn:
                host.bn_aliases.append(bn)
        else:
            ne = GlossaryEntry(en=en, bn=bn, entity_type=typ)
            entries.append(ne)
            by_en[en.lower()] = ne
        known_bn.add(bn)
    return entries


def reassign(entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
    """Expert terminology decisions that move a Bengali form between entries.

    পাতা পোড়া ("leaf burn") denotes BACTERIAL blight in Bangladesh usage, not
    blast — so detach it from Rice Blast before the additions re-attach it to
    Bacterial Blight. Keeps TermAcc from crediting the wrong disease.
    """
    for e in entries:
        if e.en == "Rice Blast":
            e.bn_aliases = [a for a in e.bn_aliases if a != "পাতা পোড়া"]
        if e.en == "Bacterial Blight":
            # "blight" alone is too broad (matches sheath/late blight); require
            # the qualified form so TermAcc applicability is precise.
            e.en_aliases = [a for a in e.en_aliases if a.lower() != "blight"]
    return entries


def export(db_path: Path | str = DEFAULT_KG_DB) -> int:
    entries = merge_additions(reassign(load_glossary(db_path)))
    GLOSSARY_TSV.parent.mkdir(parents=True, exist_ok=True)

    with GLOSSARY_TSV.open("w", encoding="utf-8", newline="\n") as f:
        f.write("en\tbn\ttype\ten_aliases\tbn_aliases\n")
        for e in sorted(entries, key=lambda x: (x.entity_type, x.en.lower())):
            f.write(
                f"{e.en}\t{e.bn}\t{e.entity_type}\t"
                f"{'|'.join(e.en_aliases)}\t{'|'.join(e.bn_aliases)}\n"
            )

    GLOSSARY_JSON.write_text(
        json.dumps(
            [
                {
                    "en": e.en,
                    "bn": e.bn,
                    "type": e.entity_type,
                    "en_aliases": e.en_aliases,
                    "bn_aliases": e.bn_aliases,
                }
                for e in entries
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(entries)


if __name__ == "__main__":
    n = export()
    # Avoid printing Bengali to a cp1252 Windows console; report counts only.
    by_type: dict[str, int] = {}
    for e in merge_additions(reassign(load_glossary())):
        by_type[e.entity_type] = by_type.get(e.entity_type, 0) + 1
    print(f"Exported {n} glossary entries -> {GLOSSARY_TSV.name}, {GLOSSARY_JSON.name}")
    print("By type:", dict(sorted(by_type.items())))
