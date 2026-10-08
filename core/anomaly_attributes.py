"""Profile-driven multilabel attributes and explicit review states."""
import re

STATES = {'unreviewed', 'reviewed', 'unknown', 'partial'}
CONTROLS = {'normal', 'unreviewed', 'unknown'}

def key(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')

def tokens(value):
    if isinstance(value, dict):
        value = value.get('anomaly_labels', value.get('labels', value.get('anomaly_label', value.get('label', ''))))
    if isinstance(value, (list, tuple, set)):
        return [t for x in value for t in tokens(x)]
    return [key(x) for x in re.split(r'[,;|]+', str(value or '')) if key(x)]

def normalize(value, profile):
    options = [key(x) for x in profile.get('anomaly_labels', []) if key(x) not in CONTROLS]
    aliases = {key(k):key(v) for k,v in profile.get('anomaly_aliases', {}).items()}
    found = list(dict.fromkeys(aliases.get(t,t) for t in tokens(value)))
    labels = [x for x in options if x in found]
    unknown = 'unknown' in found or any(x not in options and x not in CONTROLS for x in found)
    if labels:
        marker = ['unknown'] if unknown else ['unreviewed'] if 'unreviewed' in found else []
        return '; '.join(labels + marker)
    if unknown:
        return 'unknown'
    return 'normal' if 'normal' in found else 'unreviewed'

def export_review(value, profile):
    value = normalize(value, profile)
    ts = tokens(value)
    labels = [x for x in ts if x not in CONTROLS]
    state = 'unreviewed' if 'unreviewed' in ts else 'partial' if labels and 'unknown' in ts else 'unknown' if 'unknown' in ts else 'reviewed'
    return {'anomaly_labels':labels, 'anomaly_review_state':state}

def display_value(event, profile):
    if 'anomaly_labels' not in event:
        return normalize(event.get('anomaly_label', ''), profile)
    labels = list(event.get('anomaly_labels') or [])
    state = event.get('anomaly_review_state', 'unreviewed')
    if state in ('unknown', 'partial'):
        labels.append('unknown')
    elif state == 'unreviewed' and labels:
        labels.append('unreviewed')
    if not labels:
        labels = ['normal' if state == 'reviewed' else 'unknown' if state == 'unknown' else 'unreviewed']
    return normalize(labels, profile)

def is_positive(value, profile):
    record = export_review(value,profile)
    return bool(record['anomaly_labels'] and record['anomaly_review_state'] in ('reviewed','partial'))
