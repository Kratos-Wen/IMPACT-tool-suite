"""Resolve registry suffixes without erasing numeric parts of a noun class."""
import re
from functools import lru_cache
from core.annotation_migration import canonical_noun


def _normalize(value):
    return re.sub(r'[ -]+', '_', str(value or '').strip()).casefold()


@lru_cache(maxsize=64)
def _categories(noun_classes, aliases, extra_classes):
    profile={'noun_aliases':dict(aliases)}
    classes=set(noun_classes)|set(profile['noun_aliases'].values())|set(extra_classes)|{'assembly'}
    classes={_normalize(canonical_noun(_normalize(c),profile)) for c in classes if isinstance(c,str) and c}
    return tuple(sorted(classes,key=lambda c:(-len(c),c)))


@lru_cache(maxsize=4096)
def _resolve_category(label, classes):
    for category in classes:
        if label==category or (label.startswith(category) and re.fullmatch(r'(?:_\d+)+',label[len(category):])):
            return category
    return re.sub(r'(?:_\d+)+$','',label)


def instance_category(name, profile, extra_classes=()):
    # Content keys also invalidate after in-place profile/class-map edits.
    aliases=tuple(profile.get('noun_aliases',{}).items())
    classes=_categories(tuple(profile.get('noun_classes',[])),aliases,tuple(extra_classes))
    label=_normalize(canonical_noun(_normalize(name),profile))
    return _resolve_category(label,classes)
