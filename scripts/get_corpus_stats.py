import json
from pathlib import Path
import argparse

def count_words(text: str) -> int:
    return len(text.split())

def main():
    parser = argparse.ArgumentParser(description="Extract corpus statistics for the paper.")
    parser.add_argument("--corpus_dir", type=str, default="results/agribanglat5/data/corpus")
    parser.add_argument("--glossary", type=str, default="data/glossary.json")
    parser.add_argument("--pdf_dir", type=str, default="F:/Projects/Agri_bot/data")
    args = parser.parse_args()

    train_file = Path(args.corpus_dir) / "agrienbn.train.jsonl"
    dev_file = Path(args.corpus_dir) / "agrienbn.dev.jsonl"
    glossary_file = Path(args.glossary)

    en_words = 0
    bn_words = 0
    sents = 0

    if train_file.exists() and dev_file.exists():
        for f in [train_file, dev_file]:
            for line in f.read_text('utf-8').splitlines():
                if not line.strip(): continue
                d = json.loads(line)
                en_words += count_words(d.get('en', ''))
                bn_words += count_words(d.get('bn', ''))
                sents += 1

        print("=== Corpus Statistics ===")
        print(f"Total Sentences (Pairs): {sents:,}")
        print(f"Total English Words: {en_words:,} (Avg: {en_words/sents:.1f} per sentence)")
        print(f"Total Bengali Words: {bn_words:,} (Avg: {bn_words/sents:.1f} per sentence)")
    else:
        print(f"Corpus files not found in {args.corpus_dir}")

    if glossary_file.exists():
        glossary = json.loads(glossary_file.read_text('utf-8'))
        print(f"Unique Agricultural Terms (Glossary): {len(glossary)}")
    
    en_pdfs = list((Path(args.pdf_dir) / "pdfs").glob("*.pdf"))
    bn_pdfs = list((Path(args.pdf_dir) / "pdfs_bangla_pending").glob("*.pdf"))
    
    if en_pdfs or bn_pdfs:
        print(f"Source Documents: {len(en_pdfs)} English PDFs, {len(bn_pdfs)} Bengali PDFs (Total: {len(en_pdfs) + len(bn_pdfs)})")
    
if __name__ == '__main__':
    main()
