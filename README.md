# Contract RAG — page-cited Q&A over commercial contracts

Ask a question about 10 real commercial contracts and get an answer that cites the exact **file and page** it came from — or an explicit *"I could not find this in the documents"* when the answer isn't there.

The main focus of the project is **measuring retrieval**, not just building a demo: a hand-checked golden set of 38 questions was used to compare four retrieval strategies.

## Results

Hit@5 = share of questions where the page containing the answer is among the 5 retrieved pages (33 answerable questions).

| Retrieval method | Hit@5 |
|---|---|
| Vector search (`all-MiniLM-L6-v2`) | 17/33 = **52%** |
| BM25 keyword search | 22/33 = **67%** |
| **Hybrid (vector + BM25, Reciprocal Rank Fusion)** | **25/33 = 76%** |
| Hybrid + LLM query expansion (Claude Haiku) | 24–25/33 = 73–76% (varies per run) |

### What I learned

1. **Keyword search beats a small embedding model on legal text.** The decisive information in contracts sits in exact phrases ("governed by", "per occurrence", "thirty (30) days"). BM25 alone outperformed vector search by 15 points.
2. **Hybrid search was the best trade-off.** Fusing both rankings with RRF found pages that neither method found alone, with no extra cost or latency. It is the default in `ask.py`.
3. **Query expansion was not worth it — a negative result.** Rewriting the question into contract vocabulary fixed vocabulary-mismatch cases (e.g. asking for the *term* when the contract says *Agreement Period*), but broke a similar number of questions that hybrid already answered. It also adds an LLM call per query and makes results non-deterministic. On the first 11-question version of the eval set it looked like a +8 point win; on 33 questions the gain disappeared — a reminder of why the eval set had to grow.
4. **Most failures are retrieval failures, not generation failures.** When the right page was not retrieved, the model said it could not find the answer instead of guessing. All 5 trap questions (answers absent from the documents) were correctly declined, including a deliberately misleading one: asked for Agape's *Singapore* address, the model reported that the only Agape address is in Hong Kong instead of passing it off as the answer. One caveat: in two trap cases the relevant contract's pages were barely retrieved, so the model declined correctly but without having seen the whole contract.

### Remaining failure modes

| Failure | Example |
|---|---|
| Company name missing from the text | The *Loha* contract's body is written between Chinese trading companies, so "Loha" barely appears in it (q19, q20) |
| Near-duplicate documents | Two Reynolds contracts with the same governing law — the wrong one ranks higher (q07) |
| Tables of contents | Upjohn's TOC lists every clause title and outranks the actual clause on p.76 (q09) |
| Clause inside an attachment | ExxonMobil standard terms appended to the West Pharma agreement (q23) |

Next steps that target these: prepending document/section context to each chunk ("contextual chunk headers"), filtering TOC pages, and a reranker.

## How it works

```
PDF ──► ingest.py ──► pages.jsonl ──► chunk.py ──► chunks.jsonl ──► embed.py ──► Chroma
       (page-by-page     {file, page, text}   (1000 chars, 200 overlap,          (vectors)
        text extraction)                        never crosses a page)
                                                         │
question ──► retrieve.py: vector + BM25 ──► RRF ──► top-5 unique pages ──► ask.py ──► Claude
                                                                              answer with [file, p.X] citations
```

- **Page-level provenance:** text is extracted per page and chunks never span two pages, so every citation points to one real page.
- **Grounded prompting:** the model must cite `[file, p.X]` after each claim, include a short verbatim quote, and decline when the sources don't contain the answer.

## Data

10 contracts from the [CUAD dataset](https://www.atticusprojectai.org/cuad) (real agreements filed with the SEC): 4 supply, 2 service, 2 manufacturing, 1 outsourcing, 1 maintenance — 317 pages, 1,145 chunks.

The evaluation set (`eval/questions.jsonl`) has 33 answerable questions with the expected file and page, plus 5 trap questions whose answers are not in the documents. Each expected page was verified to contain the supporting text.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

python src/ingest.py           # PDFs -> data/processed/pages.jsonl
python src/chunk.py            # pages -> data/processed/chunks.jsonl
python src/embed.py            # chunks -> chroma_db/
python src/ask.py "Which law governs the Agape ATP supply agreement?"
python src/eval_retrieval.py   # compare retrieval methods
```

## Limitations

- 33 questions is still a small eval set; differences of 1–2 questions are within noise.
- Only retrieval is scored automatically; answer quality and the 5 trap questions were checked manually.
- Some contract text is redacted (`[***]`) in the source filings.


## Kaggle URL
https://www.kaggle.com/code/oguzhan18m/notebook0fe62d6312
