"""Recover Unicode Bengali from legacy Bijoy/SutonnyMJ-encoded agronomy PDFs.

AgriBot's Bangladesh-specific source documents (data/pdfs_bangla_pending) store
Bengali in the ASCII-mapped *SutonnyMJ* (Bijoy) font: a normal text extractor
returns Latin gibberish, not Bengali. We recover real Unicode by:

  1. extracting text span-by-span with its font (PyMuPDF),
  2. converting ONLY Bijoy-font spans with bijoy2unicode, leaving genuine
     English/Unicode spans (Times/Arial/Calibri/Nikosh) untouched,
  3. normalizing the Bengali (bnunicodenormalizer),
  4. segmenting into sentences and keeping those with enough Bengali content.

Output: one recovered text file per PDF + a pooled sentence file used as the
Bangladesh-specific seed for back-translation and terminology mining.

    python -m agri_mt.recover_bijoy
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

# Font families that are ASCII-mapped Bijoy (need conversion). Anything else
# (TimesNewRoman, Arial, Calibri, MyriadPro, Nikosh=Unicode) is left as-is.
BIJOY_FONT_PREFIXES = ("SutonnyMJ", "SutonnyOMJ", "Sutonny")

BENGALI_RE = re.compile(r"[ঀ-৿]")
DEFAULT_SRC = Path(r"F:/Projects/Agri_bot/data/pdfs_bangla_pending")
HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "data" / "recovered"
MIN_BENGALI_CHARS = 12  # a "sentence" must carry at least this many Bengali chars


def _converter():
    from bijoy2unicode import converter

    return converter.Unicode()


def _is_bijoy(font: str) -> bool:
    return any(font.startswith(p) for p in BIJOY_FONT_PREFIXES)


def _normalizer():
    try:
        from bnunicodenormalizer import Normalizer

        return Normalizer()
    except Exception:
        return None


def extract_recovered_text(pdf_path: Path, conv, norm) -> str:
    """Return the page text of a PDF with Bijoy spans converted to Unicode."""
    doc = fitz.open(str(pdf_path))
    out_lines: list[str] = []
    for page in doc:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                parts: list[str] = []
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    if _is_bijoy(span.get("font", "")):
                        try:
                            text = conv.convertBijoyToUnicode(text)
                        except Exception:
                            pass  # keep raw on rare converter failure
                    parts.append(text)
                line_text = "".join(parts).strip()
                if line_text:
                    out_lines.append(line_text)
    doc.close()
    text = "\n".join(out_lines)
    if norm is not None:
        # Normalize Bengali token-wise; leave non-Bengali tokens alone.
        text = _normalize_bengali_words(text, norm)
    return text


def _normalize_bengali_words(text: str, norm) -> str:
    def fix(tok: str) -> str:
        if BENGALI_RE.search(tok):
            try:
                r = norm(tok)
                return r["normalized"] if isinstance(r, dict) and r.get("normalized") else tok
            except Exception:
                return tok
        return tok

    return " ".join(fix(t) for t in text.split(" "))


def segment_sentences(text: str) -> list[str]:
    """Split on Bengali danda and ./!/? ; keep Bengali-bearing sentences."""
    parts = re.split(r"[।\.\!\?\n]+", text)
    out: list[str] = []
    for p in parts:
        s = re.sub(r"\s+", " ", p).strip()
        if len(BENGALI_RE.findall(s)) >= MIN_BENGALI_CHARS:
            out.append(s)
    return out


def recover_all(src_dir: Path = DEFAULT_SRC) -> dict[str, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conv, norm = _converter(), _normalizer()
    pooled: list[str] = []
    counts: dict[str, int] = {}
    for pdf in sorted(src_dir.glob("*.pdf")):
        text = extract_recovered_text(pdf, conv, norm)
        (OUT_DIR / f"{pdf.stem}.bn.txt").write_text(text, encoding="utf-8")
        sents = segment_sentences(text)
        counts[pdf.name] = len(sents)
        pooled.extend(sents)
    # Deduplicate pooled sentences, preserve order.
    seen: dict[str, None] = {}
    for s in pooled:
        seen.setdefault(s, None)
    (OUT_DIR / "all_bn_sentences.txt").write_text("\n".join(seen), encoding="utf-8")
    counts["_TOTAL_sentences"] = sum(v for k, v in counts.items())
    counts["_UNIQUE_sentences"] = len(seen)
    return counts


if __name__ == "__main__":
    c = recover_all()
    print("Recovered Bengali sentences per file:")
    for k, v in c.items():
        if not k.startswith("_"):
            print(f"  {v:5d}  {k}")
    print(f"TOTAL (with dups): {c['_TOTAL_sentences']}")
    print(f"UNIQUE: {c['_UNIQUE_sentences']}  -> data/recovered/all_bn_sentences.txt")
