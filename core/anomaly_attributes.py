"""Profile-driven anomaly types. An empty selection is normal."""
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
    return '; '.join(labels) if labels else 'normal'

def export_review(value, profile):
    value = normalize(value, profile)
    ts = tokens(value)
    labels = [x for x in ts if x not in CONTROLS]
    # This field records a resolved classification, not human track verification.
    return {'anomaly_labels':labels, 'anomaly_review_state':'reviewed'}

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
