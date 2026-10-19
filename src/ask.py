"""
Adım 4 — Soru-cevap
Çalıştırma:  python src/ask.py "What is the term of the West Pharmaceutical supply agreement?"
"""
import sys
from retrieve import retrieve
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-5-5"   # hızlı ve ucuz; ileride Sonnet ile karşılaştırabilirsin
TOP_K = 5

SYSTEM = """You answer questions about contracts using ONLY the provided sources.
- After every claim, cite the source as [file, p.X].
- Include a short verbatim quote from the source supporting the answer.
- If the sources do not contain the answer, say "I could not find this in the documents." Do not guess."""


def build_context(hits):
    parts = []
    for c in hits:
        parts.append(f"[{c['file']}, p.{c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    question = sys.argv[1]
    hits = retrieve(question)
    context = build_context(hits)

    client = anthropic.Anthropic()   # anahtarı ANTHROPIC_API_KEY'den kendisi okur
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"}],
    )

    answer = "".join(b.text for b in msg.content if b.type == "text")
    print("CEVAP:\n" + answer)
    print("\nGETİRİLEN KAYNAKLAR:")
    for c in hits:
        print(f"  - {c['file']}, s.{c['page']}")