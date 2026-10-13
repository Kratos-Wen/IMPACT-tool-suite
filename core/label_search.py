"""Small-vocabulary substring and typo suggestions; labels are never auto-renamed."""
from difflib import SequenceMatcher
import re


def ranked_labels(query, labels, limit=12):
    def normalize(text):
        return re.sub(r'[\s_-]+', '_', re.sub(r'^\[\d+\]\s*', '', str(text).strip()).casefold())
    query = normalize(query)
    if query.isdigit():
        return []
    candidates = list(dict.fromkeys(str(label) for label in labels if str(label).strip()))
    if not query:
        return candidates[:limit]
    scored = []
    for label in candidates:
        name = normalize(label)
        if query == name:
            score = 3.0
        elif query in name:
            score = 2.0 + len(query) / max(1, len(name))
        elif len(query) >= 3:
            score = SequenceMatcher(None, query, name).ratio()
            if score < 0.65:
                continue
        else:
            continue
        scored.append((score, label))
    return [label for score, label in sorted(scored, key=lambda row: (-row[0], row[1]))[:limit]]
