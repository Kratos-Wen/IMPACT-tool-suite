import re
from core.project_profile import PROFILE
_ALIASES={re.sub(r'[ -]+','_',str(k).strip()).casefold():str(v) for k,v in (PROFILE.get('noun_aliases') or {}).items()}
_FIELDS={'label','category','noun','target','tool','instrument','object_name','class_name','class_label','noun_class'}
_MAPS={'class_map','id_to_category','names'}
_LISTS={'category_candidates','noun_library','object_classes','classes','names','components'}
_RAW={'raw_response','raw_model_output','machine_original','machine_snapshot','raw_snapshot','source_record'}
def _canonical(value, instance=False):
 if not isinstance(value,str):return value
 text=re.sub(r'[ -]+','_',value.strip())
 key=text.casefold()
 if key in _ALIASES:return _ALIASES[key]
 match=re.fullmatch(r'(.*)_(\d+)',text)
 if instance and match and match[1].casefold() in _ALIASES:return _ALIASES[match[1].casefold()]+'_'+match[2]
 return value

def normalize_noun_aliases(value,path='',parent=''):
 """Normalise confirmed class aliases at import, preserving IDs and raw provenance."""
 if isinstance(value,list):
  result=[normalize_noun_aliases(v,path+'/'+str(i),parent) for i,v in enumerate(value)]
  if parent in _LISTS or '/hoi_ontology/relations/' in path:
   result=[_canonical(v) for v in result]
   if (parent in {'category_candidates','noun_library'} or '/hoi_ontology/relations/' in path) and all(isinstance(v,str) for v in result):result=list(dict.fromkeys(result))
  return result
 if not isinstance(value,dict):return value
 result={}
 for key,item in value.items():
  location=path+'/'+str(key)
  if key in _RAW:result[key]=item;continue
  if isinstance(item,str) and (key in _FIELDS or parent in _MAPS):result[key]=_canonical(item,key in {'label','object_name'} and '/object_library/' in location)
  else:result[key]=normalize_noun_aliases(item,location,str(key))
 return result


def refresh_aliases():
 _ALIASES.clear()
 _ALIASES.update({re.sub(r'[ -]+','_',str(k).strip()).casefold():str(v) for k,v in (PROFILE.get('noun_aliases') or {}).items()})
