"""Resolve registry suffixes without erasing numeric parts of a noun class."""
import re
from core.annotation_migration import canonical_noun


def instance_category(name, profile, extra_classes=()):
    def normalize(value):
        return re.sub(r'[ -]+', '_', str(value or '').strip()).casefold()
    label=normalize(canonical_noun(normalize(name),profile))
    classes=set(profile.get('noun_classes',[]))|set(profile.get('noun_aliases',{}).values())|set(extra_classes)|{'assembly'}
    classes={normalize(canonical_noun(normalize(c),profile)) for c in classes if isinstance(c,str) and c}
    for category in sorted(classes,key=lambda c:(-len(c),c)):
        if label==category or re.fullmatch(re.escape(category)+r'(?:_\d+)+',label):
            return category
    return re.sub(r'(?:_\d+)+$','',label)
