"""
Adım 3 — Embed + Chroma
Girdi : data/processed/chunks.jsonl
Çıktı : chroma_db/  (vektör veritabanı, diskte)

Çalıştırma:  python src/embed.py
"""
import json
import chromadb

IN_PATH = "data/processed/chunks.jsonl"
DB_PATH = "chroma_db"
COLLECTION = "contracts"

if __name__ == "__main__":
    with open(IN_PATH) as f:
        chunks = [json.loads(line) for line in f]

    client = chromadb.PersistentClient(path=DB_PATH)   # diske kaydeden veritabanı
    col = client.get_or_create_collection(COLLECTION)  # tablo gibi düşün

    # Chroma 3 paralel liste ister: aynı sıradaki elemanlar aynı parçaya ait
    # TODO 1: ids        -> her parçanın chunk_id'si          [c["chunk_id"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    # TODO 2: documents  -> her parçanın metni                 (aynı kalıp, "text")
    documents = [c["text"] for c in chunks]
    # TODO 3: metadatas  -> her parçanın dosya ve sayfa bilgisi [{"file": c["file"], "page": c["page"]} for c in chunks]
    metadatas = [{"file": c["file"], "page": c["page"]} for c in chunks]

    # upsert = varsa güncelle, yoksa ekle (dosyayı tekrar çalıştırınca kopya oluşmaz)
    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Veritabanında {col.count()} parça var.")

    # Hızlı test: bir soru sor, en yakın 3 parçayı göster
    q = "What are the payment terms?"
    res = col.query(query_texts=[q], n_results=3)
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print(f"\n[{meta['file']}, s.{meta['page']}]\n{doc[:200]}")