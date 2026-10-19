"""
Retrieval katmanı: vektör arama, BM25 (anahtar kelime) arama ve ikisinin birleşimi (hybrid).
Diğer scriptler buradan import eder:  from retrieve import retrieve
"""
import json
import re
import chromadb
from rank_bm25 import BM25Okapi

DB_PATH = "chroma_db"
COLLECTION = "contracts"
CHUNKS_PATH = "data/processed/chunks.jsonl"
RRF_K = 60   # RRF formülündeki sabit; standart değer


def tokenize(text):
    """Metni küçük harfli kelimelere böl: 'Governed by Illinois' -> ['governed', 'by', 'illinois']"""
    return re.findall(r"\w+", text.lower())


# --- Program açılırken bir kez hazırlanan şeyler ---
_col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
with open(CHUNKS_PATH) as f:
    _chunks = [json.loads(line) for line in f]
_by_id = {c["chunk_id"]: c for c in _chunks}
_bm25 = BM25Okapi([tokenize(c["text"]) for c in _chunks])


def vector_search(question, n):
    """Anlamca en yakın n parçanın id'leri (en iyiden kötüye)."""
    res = _col.query(query_texts=[question], n_results=n)
    return res["ids"][0]


def bm25_search(question, n):
    """Sorudaki kelimeleri en çok içeren n parçanın id'leri (en iyiden kötüye)."""
    scores = _bm25.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [_chunks[i]["chunk_id"] for i in top]


def hybrid_search(question, n):
    """
    İki listeyi Reciprocal Rank Fusion (RRF) ile birleştirir.
    Her parça, her listedeki SIRASINA göre puan alır:  1 / (RRF_K + sıra)
    İki listede de üst sıralarda olan parça en yüksek toplam puanı alır.
    """
    scores = {}
    for ranked in (vector_search(question, n), bm25_search(question, n)):
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


def dedupe_pages(chunk_ids, k):
    """Aynı sayfadan gelen tekrarları at, ilk k farklı sayfanın parçasını döndür."""
    seen, out = set(), []
    for cid in chunk_ids:
        c = _by_id[cid]
        key = (c["file"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) == k:
            break
    return out


def retrieve(question, k=5, method="hybrid"):
    """method: 'vector' | 'bm25' | 'hybrid'. Her parça {file, page, chunk_id, text} şeklinde döner."""
    search = {"vector": vector_search, "bm25": bm25_search, "hybrid": hybrid_search}[method]
    pool = k * 4   # tekrarlar atılınca k'nın altına düşmemek için fazladan getir
    return dedupe_pages(search(question, pool), k)
