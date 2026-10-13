"""Resolve registry suffixes without erasing numeric parts of a noun class."""
import re
from functools import lru_cache
from core.annotation_migration import canonical_noun


def _normalize(value):
    return re.sub(r'[ -]+', '_', str(value or '').strip()).casefold()

@lru_cache(maxsize=128)
def _known_categories(noun_classes, aliases, extra_classes):
    profile = {'noun_aliases': dict(aliases)}
    classes=set(noun_classes)|set(profile['noun_aliases'].values())|set(extra_classes)|{'assembly'}
    normalize = _normalize
    classes={normalize(canonical_noun(normalize(c),profile)) for c in classes if isinstance(c,str) and c}
    return tuple(sorted(classes,key=lambda c:(-len(c),c)))

@lru_cache(maxsize=4096)
def _match_category(label, classes):
    for category in classes:
        if label==category or re.fullmatch(re.escape(category)+r'(?:_\d+)+',label):
            return category
    return re.sub(r'(?:_\d+)+$','',label)

def instance_category(name, profile, extra_classes=()):
    label=_normalize(canonical_noun(_normalize(name),profile))
    classes=_known_categories(tuple(profile.get('noun_classes',[])),
                              tuple(profile.get('noun_aliases',{}).items()),tuple(extra_classes))
    return _match_category(label,classes)
