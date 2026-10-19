"""
Adım 2 — Chunking
Girdi : data/processed/pages.jsonl   (her satır bir sayfa)
Çıktı : data/processed/chunks.jsonl  (her satır bir parça)

Her parça şu yapıda olacak:
    {"file": "01_supply_agape.pdf", "page": 5, "chunk_id": "01_supply_agape.pdf-p5-c0", "text": "..."}

Çalıştırma:  python src/chunk.py
"""
import json

IN_PATH = "data/processed/pages.jsonl"
OUT_PATH = "data/processed/chunks.jsonl"

CHUNK_SIZE = 1000   # bir parçanın en fazla kaç karakter olacağı (~250 token)
OVERLAP = 200       # art arda iki parçanın kaç karakter ortak olacağı


def clean_text(text):
    """CUAD'ın her sayfaya eklediği 'Source: ...' satırını çıkar."""
    lines = text.splitlines()
    kept = [line for line in lines if not line.startswith("Source:")]
    return "\n".join(kept).strip()


def split_text(text, size, overlap):
    """Metni 'size' uzunluğunda, 'overlap' kadar örtüşen parçalara böl."""
    chunks = []
    start = 0
    step = size - overlap
    
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return chunks


if __name__ == "__main__":
    with open(IN_PATH) as f:
        pages = [json.loads(line) for line in f]

    all_chunks = []
    for p in pages:
        text = clean_text(p["text"])
        for i, chunk_text in enumerate(split_text(text, CHUNK_SIZE, OVERLAP)):
            all_chunks.append({
                "file": p["file"],
                "page": p["page"],
                "chunk_id": f'{p["file"]}-p{p["page"]}-c{i}',
                "text": chunk_text,
            })

    with open(OUT_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"{len(pages)} sayfa -> {len(all_chunks)} parça -> {OUT_PATH}")
    # Kontrol: ilk parçayı göster
    print(all_chunks[0])
