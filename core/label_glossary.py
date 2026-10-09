"""Project-owned label explanations; this module contains no dataset policy."""
import re
from core.project_profile import PROFILE


def category_for_label(label, kind, profile=None):
    profile = PROFILE if profile is None else profile
    text = re.sub(r"^\[\d+\]\s*", "", str(label or "").strip())
    aliases = profile.get("noun_aliases" if kind == "noun" else "verb_aliases", {})
    definitions = profile.get(kind + "_explanations", {})
    if text not in definitions and text not in aliases:
        text = re.sub(r'[ -]+', '_', text).casefold()
    # Exact categories take precedence over stripping a physical instance suffix.
    if text not in definitions and text not in aliases and kind == "noun":
        text = re.sub(r"_\d+$", "", text)
    return aliases.get(text, text)


def explanation(label, kind, language="en", profile=None):
    profile = PROFILE if profile is None else profile
    category = category_for_label(label, kind, profile)
    entry = profile.get(kind + "_explanations", {}).get(category)
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        return str(entry.get(language) or entry.get("en") or "").strip()
    return ""


def label_tooltip(label, kind):
    text = explanation(label, kind)
    return text or "Added label: agree on its definition with the coordinator."


def glossary_rows(kind, extra=(), language="en"):
    names = PROFILE.get("noun_classes" if kind == "noun" else "verbs", [])
    categories = {category_for_label(name, kind) for name in [*names, *extra] if name}
    return [(name, explanation(name, kind, language) or
             "Added label: agree on its definition with the coordinator.")
            for name in sorted(categories)]
