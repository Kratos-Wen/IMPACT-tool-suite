"""Large alias tables stay fast and cache keys follow in-place profile changes."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import annotation_migration as migration
from core.instance_category import instance_category
profile=dict(noun_classes=['part','tool_15','tool_20'],noun_aliases={f'old_part_{i}':'part' for i in range(60)})
profile['noun_aliases'].update({'old tool':'tool_15','OLD-TOOL':'tool_20'})
migration._normalized_noun_aliases.cache_clear()
with patch.object(migration,'key',wraps=migration.key) as normalize:
 for _ in range(1000):assert migration.canonical_noun('old_part_10',profile)=='part'
 assert normalize.call_count<1100,'Alias normalization scaled with every lookup'
assert migration.canonical_noun('OLD-TOOL',profile)=='tool_20'
assert instance_category('tool_15_2',profile)=='tool_15'
assert instance_category('tool_20',profile)=='tool_20'
assert instance_category('old_part_10_2',profile)=='part'
profile['noun_aliases']['old_part_10']='attachment'
profile['noun_classes'].append('attachment')
assert migration.canonical_noun('old_part_10',profile)=='attachment'
assert instance_category('old_part_10_2',profile)=='attachment'
assert instance_category('custom_15',profile,['custom_15'])=='custom_15'
assert instance_category('custom_15',profile)=='custom'
print('ALIAS_1000_LOOKUPS_PROFILE_MUTATION_NUMERIC_CLASS_AND_EXTRA_CLASS_INVALIDATION_PASS')
