"""
Bir ifadenin hangi dosya ve sayfada geçtiğini bulur (soru seti hazırlamak için).
Kullanım:  python src/find.py "term" 04_supply_west_pharma
"""
import sys
import json

phrase = sys.argv[1].lower()
file_filter = sys.argv[2] if len(sys.argv) > 2 else ""

with open("data/processed/pages.jsonl") as f:
    for line in f:
        p = json.loads(line)
        if file_filter not in p["file"]:
            continue
        text = p["text"].lower()
        i = text.find(phrase)
        if i != -1:
            snippet = p["text"][max(0, i - 80): i + 150].replace("\n", " ")
            print(f"[{p['file']}, s.{p['page']}]  ...{snippet}...\n")