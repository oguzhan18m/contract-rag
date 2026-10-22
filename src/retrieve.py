"""
Retrieval katmanı: vektör arama, BM25 (anahtar kelime) arama ve ikisinin birleşimi (hybrid).
Diğer scriptler buradan import eder:  from retrieve import retrieve
"""
import json
import re
import chromadb
import anthropic
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

DB_PATH = "chroma_db"
COLLECTION = "contracts"
CHUNKS_PATH = "data/processed/chunks.jsonl"
RRF_K = 60   # RRF formülündeki sabit; standart değer
EXPAND_MODEL = "claude-haiku-5-5"


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


EXPAND_PROMPT = """You help search a collection of commercial contracts.
Rewrite the user's question as 3 alternative search queries that use wording
likely to appear in the contract text itself (legal synonyms, section headings,
typical clause phrasing). Keep any company names.
Return exactly 3 lines, one query per line, no numbering, no extra text."""


def expand_query(question):
    """Claude'dan sorunun sözleşme diliyle yazılmış 3 alternatifini iste."""
    msg = anthropic.Anthropic().messages.create(
        model=EXPAND_MODEL,
        max_tokens=300,
        system=EXPAND_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return [line.strip() for line in text.splitlines() if line.strip()][:3]


def expanded_search(question, n):
    """
    Orijinal soru + 3 alternatif = 4 sorgu. Her sorgu için vektör ve BM25 listesi alınır
    (toplam 8 liste) ve hepsi hybrid_search'teki gibi RRF ile birleştirilir.
    """
    queries = [question] + expand_query(question)
    scores = {}
    # TODO: queries listesindeki HER sorgu için, hybrid_search'teki iki döngünün aynısını çalıştır
    #   ipucu: en dışa  "for q in queries:"  ekle; içeride vector_search(q, n) ve bm25_search(q, n) kullan
    for q in queries:
        for ranked in (vector_search(q, n), bm25_search(q, n)):
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
    """method: 'vector' | 'bm25' | 'hybrid' | 'expanded'. Her parça {file, page, chunk_id, text} şeklinde döner."""
    search = {"vector": vector_search, "bm25": bm25_search,
              "hybrid": hybrid_search, "expanded": expanded_search}[method]
    pool = k * 4   # tekrarlar atılınca k'nın altına düşmemek için fazladan getir
    return dedupe_pages(search(question, pool), k)
