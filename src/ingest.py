"""
Adım 1 — Ingest
Amaç: Bir PDF'i sayfa sayfa okuyup her sayfa için şu kaydı üretmek:
    {"file": "01_supply_agape.pdf", "page": 1, "text": "..."}

Çalıştırma:  python src/ingest.py
"""
import os
import json
import pymupdf

PDF_PATH = "data/raw/01_supply_agape.pdf"


def read_pdf(path):
    pages = []
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        pages.append({
            "file": os.path.basename(path),
            "page": i + 1,
            "text": page.get_text(),
        })
    doc.close()

    return pages


if __name__ == "__main__":
    RAW_DIR = "data/raw"
    OUT_PATH = "data/processed/pages.jsonl"

    all_pages = []
    for name in sorted(os.listdir(RAW_DIR)):
        if not name.endswith(".pdf"):
            continue
        path = os.path.join(RAW_DIR, name)
        pages = read_pdf(path)
        all_pages.extend(pages)
        print(f"{name}: {len(pages)} sayfa")

    with open(OUT_PATH, "w") as f:
        for p in all_pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Toplam {len(all_pages)} sayfa -> {OUT_PATH}")
