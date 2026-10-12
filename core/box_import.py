"""Selective machine-box import; existing manual evidence is retained."""
def automatic(box, track=None):
    if box.get('human_verified') or box.get('locked'):return False
    source=str(box.get('source') or (track or {}).get('source') or '').lower()
    if source.startswith(('manual','human','materialized_handtrack')):return False
    return (any(word in source for word in ('automatic','machine','model','proposal','yolo','mediapipe','sam2','qwen','inferred')) or
            bool(box.get('automatic') or (track or {}).get('automatic')))

def import_box(box,is_hand,mode='all',track=None):
    if mode not in ('all','hands','none'):raise ValueError('Invalid automatic-box import mode')
    return not automatic(box,track) or mode=='all' or mode=='hands' and is_hand

def unique_hand_candidates(boxes,normalize_hand):
    """Keep manual evidence; omit ambiguous machine hands without guessing identity."""
    from core.frame_review import signature
    groups={};kept=[];omitted=0
    for box in boxes:
        hand=normalize_hand(box.get('label'))
        if hand:groups.setdefault((box.get('orig_frame'),hand),[]).append(box)
        else:kept.append(box)
    for rows in groups.values():
        manual=[b for b in rows if not automatic(b)]
        if manual:
            kept.extend(manual);omitted+=len(rows)-len(manual);continue
        valid={}
        for box in rows:
            try:valid.setdefault(signature([box]),box)
            except (TypeError,ValueError,KeyError):pass
        if len(valid)==1:kept.extend(valid.values());omitted+=len(rows)-1
        else:omitted+=len(rows)
    return kept,omitted


def constrained_import_mode(value, profile):
    value=value if value in ('all','hands','none') else 'all'
    limit=profile.get('automatic_tracking_import_limit','all')
    if limit=='none':return 'none'
    if limit=='hands' and value=='all':return 'hands'
    return value
