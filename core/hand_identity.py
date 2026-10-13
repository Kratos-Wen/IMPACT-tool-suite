"""Two anatomical hand entities; numeric handles stay outside ordinary object IDs."""
HAND_IDS = {'left_hand': -1, 'right_hand': -2}
HAND_ENTITIES = {'left_hand': 'H_LEFT', 'right_hand': 'H_RIGHT'}

def hand_box_id(label):
    return HAND_IDS.get(str(label or '').strip().lower())

def hand_entity_id(label):
    return HAND_ENTITIES.get(str(label or '').strip().lower())

def canonicalize_hand_box(box, normalize_hand):
    label = normalize_hand(box.get('label'))
    uid = hand_box_id(label)
    if uid is None:
        return box
    previous = box.get('id')
    if previous is not None and previous != uid:
        box.setdefault('source_record_id', previous)
    box.update(id=uid, label=label, entity_id=hand_entity_id(label))
    return box
