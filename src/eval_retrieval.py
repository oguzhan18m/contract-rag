"""
Adım 5b — Retrieval eval
Altın sorulardaki (fact) her soru için: doğru sayfa, getirilen ilk K sayfa arasında mı?
Üç yöntemi yan yana ölçer: vector, bm25, hybrid.

Çalıştırma:  python src/eval_retrieval.py
"""
import json
from retrieve import retrieve

QUESTIONS_PATH = "eval/questions.jsonl"
TOP_K = 5
METHODS = ["vector", "bm25", "hybrid", "expanded"]   # "expanded" Claude çağırır (~15 çağrı, birkaç sent)

if __name__ == "__main__":
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    facts = [q for q in questions if q["type"] == "fact"]

    scores = {}
    for method in METHODS:
        print(f"\n=== {method} ===")
        hits = 0
        for q in facts:
            got = retrieve(q["question"], TOP_K, method)
            hit = any(c["file"] == q["file"] and c["page"] == q["page"] for c in got)
            hits += hit
            mark = "✅" if hit else "❌"
            print(f"{mark} {q['id']}  " + ", ".join(f"{c['file'][:2]}/s.{c['page']}" for c in got)
                  + ("" if hit else f"   (beklenen {q['file'][:2]}/s.{q['page']})"))
        scores[method] = hits

    print(f"\n--- Hit@{TOP_K} ({len(facts)} soru) ---")
    for method, hits in scores.items():
        print(f"{method:7s}: {hits}/{len(facts)} = %{100 * hits / len(facts):.0f}")
