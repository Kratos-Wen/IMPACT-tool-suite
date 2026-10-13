"""Cached categories preserve aliases, numeric class names and live profile edits."""
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.annotation_migration import canonical_noun
from core.instance_category import instance_category, _categories
from core.anomaly_attributes import key

def reference(name,profile,extra=()):
    def canonical(value):
        value=str(value or '').strip('\ufeff').strip()
        aliases={key(a):b for a,b in profile.get('noun_aliases',{}).items()}
        normalized=key(value)
        if normalized in aliases:return aliases[normalized]
        match=re.fullmatch(r'(.*)_(\d+)',normalized)
        if match and match[1] in aliases:return str(aliases[match[1]])+'_'+match[2]
        return value
    def normalize(value):return re.sub(r'[ -]+','_',str(value or '').strip()).casefold()
    label=normalize(canonical(normalize(name)))
    classes=set(profile.get('noun_classes',[]))|set(profile.get('noun_aliases',{}).values())|set(extra)|{'assembly'}
    classes={normalize(canonical(normalize(c))) for c in classes if isinstance(c,str) and c}
    for category in sorted(classes,key=lambda c:(-len(c),c)):
        if label==category or re.fullmatch(re.escape(category)+r'(?:_\d+)+',label):return category
    return re.sub(r'(?:_\d+)+$','',label)

profile=dict(noun_classes=['tool','tool_15','part','a.b'],noun_aliases={'Old Part':'part'})
names=['tool','tool_15','tool_15_1_2','old-part_2','a.b_1','unknown_2_3','assembly_0',' Tool 15 1 ',None]
for extras in ((),('tool_15_1','new_class')):
    for name in names:
        assert instance_category(name,profile,extras)==reference(name,profile,extras)
hits=_categories.cache_info().hits
for _ in range(100):instance_category('tool_15_1_2',profile)
assert _categories.cache_info().hits>=hits+100
profile['noun_aliases']['Old Part']='replacement'
assert canonical_noun('old-part_2',profile)=='replacement_2'
assert instance_category('old-part_2',profile)=='replacement'
profile['noun_classes'].append('tool_15_1')
assert instance_category('tool_15_1_2',profile)=='tool_15_1'
profile.clear();profile.update(noun_classes=['new'],noun_aliases={'Old Part':'new'})
assert instance_category('old-part_2',profile)=='new'
assert instance_category('tool_15_1_2',profile,('tool_15',))=='tool_15'
print('NOUN_CATEGORY_EQUIVALENCE_AND_IN_PLACE_INVALIDATION_PASS')
