"""Portable, lossless human review of a frozen IMPACT machine snapshot.

No model inference, no implicit onset, no automatic semantic deduplication.
Frame indices are zero based in the frozen source; End is inclusive.
"""
import copy
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'IMPACT-PREANNOTATION-REVIEW-1.0'
from core.project_profile import PROFILE
VERBS = list(PROFILE.get("verbs") or [])
FORBIDDEN_VERBS = dict(PROFILE.get("forbidden_verbs") or {})
STATES = ['needs_review', 'reviewed', 'reviewed_unknown', 'rejected', 'duplicate']
BOUNDARIES = ('start_frame', 'onset_frame', 'end_frame')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def blank_event(frame=None):
    return dict(hand='unknown', verb='', novel_verb_definition=None,
                instrument_instance_id=None, object_instance_id=None,
                start_frame=frame, onset_frame=None, end_frame=frame,
                boundary_intervals=[], evidence=[], review_reasons=[],
                truncated_at_window_start=False, truncated_at_window_end=False,
                status='suggested')


def blank_review():
    return dict(status='needs_review', reviewer='', notes='', unknown_reason='',
                duplicate_of='', updated_at_utc=None)


def event_issues(event, frame_count, instance_ids):
    issues = []
    if event.get('hand') not in ('left', 'right', 'unknown'):
        issues.append('hand must be anatomical left/right/unknown')
    verb = str(event.get('verb') or '').strip()
    if not verb:
        issues.append('Missing verb')
    elif verb.casefold() in FORBIDDEN_VERBS:
        issues.append(FORBIDDEN_VERBS[verb.casefold()])
    elif VERBS and verb not in VERBS and not str(event.get('novel_verb_definition') or '').strip():
        issues.append('Custom verb requires a definition')
    for key in BOUNDARIES:
        v = event.get(key)
        if v is not None and (type(v) is not int or not 0 <= v < frame_count):
            issues.append(key + ' outside source frames')
    s, c, e = (event.get(k) for k in BOUNDARIES)
    if all(type(x) is int or x is None for x in (s, c, e)):
        if s is not None and e is not None and s > e:
            issues.append('Start > End')
        if c is not None and ((s is not None and c < s) or (e is not None and c > e)):
            issues.append('Onset outside event')
    i, o = event.get('instrument_instance_id'), event.get('object_instance_id')
    if i is not None and i == o:
        issues.append('Instrument and Object cannot be the same instance')
    for key in ('instrument_instance_id', 'object_instance_id'):
        if event.get(key) is not None and event[key] not in instance_ids:
            issues.append(key + ' does not exist in this trial')
    intervals = event.get('boundary_intervals', [])
    if not isinstance(intervals, list):
        issues.append('Boundary intervals must be a list')
    else:
        for b in intervals:
            if not isinstance(b, dict):
                issues.append('Invalid uncertainty interval'); continue
            a, z = b.get('first_possible_frame'), b.get('last_possible_frame')
            if b.get('boundary') not in ('start', 'onset', 'end'):
                issues.append('Unknown boundary interval kind')
            if type(a) is not int or type(z) is not int or not 0 <= a <= z < frame_count:
                issues.append('Invalid uncertainty interval frames')
    return issues


class ReviewDocument:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding='utf-8'))
        if self.data.get('schema') != SCHEMA:
            raise ValueError('请选择批次中的 review.json 或本工具保存的 reviewed.json')
        self._base_digest = self.data['base_snapshot_sha256']
        if digest(self.data['machine_snapshot']) != self._base_digest:
            raise ValueError('Machine snapshot checksum mismatch')
        self.video_path = self.resolve(self.data['video_path'])
        self.tracks_path = self.resolve(self.data['tracks_path'])
        uids = [e['event_uid'] for e in self.data['events']]
        if len(uids) != len(set(uids)):
            raise ValueError('Duplicate event_uid')

    def resolve(self, path):
        return (self.path.parent / path).resolve()

    @property
    def events(self):
        return self.data['events']

    @property
    def frame_count(self):
        return self.data['source']['expected_frames']

    def instance_ids(self):
        return set(self.data['instances']) | set(self.data['human_instances'])

    def instance(self, uid, frame):
        base = copy.deepcopy(self.data['instances'].get(uid, self.data['human_instances'].get(uid, {})))
        base.setdefault('label', ', '.join(base.get('category_candidates', [])))
        base.setdefault('anatomical_hand', base.get('anatomical_hand_candidate') or 'unknown')
        base.setdefault('identity_status', 'provisional_local_track')
        for edit in self.data['instance_edits']:
            if edit['track_uid'] == uid and edit['start_frame'] <= frame <= edit['end_frame']:
                base.update(edit)
        return base

    def add_event(self, frame, reviewer):
        row = dict(event_uid=self.data['trial_id'] + ':human:' + uuid.uuid4().hex,
                   origin='human_added', original=None, value=blank_event(frame), review=blank_review())
        row['review']['reviewer'] = reviewer
        self.events.append(row)
        return row

    def validate_row(self, row):
        review, value = row['review'], row['value']
        status = review['status']
        if status not in STATES:
            return ['Invalid review status']
        if status == 'needs_review':
            return []  # Incomplete / invalid drafts must remain saveable.
        issues = []
        if not str(review.get('reviewer', '')).strip():
            issues.append('请填写标注人编号')
        if status in ('rejected', 'duplicate'):
            if not str(review.get('notes', '')).strip():
                issues.append('拒绝／重复需要填写原因')
            if status == 'duplicate':
                target = review.get('duplicate_of')
                rows = {r['event_uid']: r for r in self.events}
                if target not in rows or target == row['event_uid']:
                    issues.append('重复事件必须指向本视频中的另一条事件 UID')
                else:
                    seen = {row['event_uid']}
                    while target in rows:
                        if target in seen:
                            issues.append('重复引用形成循环'); break
                        seen.add(target)
                        target = rows[target]['review'].get('duplicate_of') if rows[target]['review']['status'] == 'duplicate' else None
            return issues
        issues.extend(event_issues(value, self.frame_count, self.instance_ids()))
        unknown = ([k for k in BOUNDARIES if value.get(k) is None]
                   + (['hand'] if value.get('hand') == 'unknown' else [])
                   + (['object_instance_id'] if value.get('object_instance_id') is None else []))
        if unknown and status != 'reviewed_unknown':
            issues.append('仍有未知项；请用“已审核但无法确定”并填写原因：' + ', '.join(unknown))
        if status == 'reviewed_unknown' and not str(review.get('unknown_reason', '')).strip():
            issues.append('未知项需要说明原因，不能补成 Start')
        return issues

    def validate(self):
        errors = []
        if digest(self.data['machine_snapshot']) != self._base_digest:
            errors.append('Original machine snapshot was modified')
        for row in self.events:
            errors.extend(row['event_uid'] + ': ' + issue for issue in self.validate_row(row))
        return errors

    def completion(self):
        events = self.events
        windows = self.data['windows']
        return dict(events_total=len(events), events_resolved=sum(r['review']['status'] != 'needs_review' for r in events),
                    windows_total=len(windows), windows_replayed=sum(w['review']['status'] == 'replayed' for w in windows),
                    geometry_fully_reviewed=False, anomaly_annotation_in_scope=False,
                    final_ground_truth_certified=False)

    def save(self, path, reviewer):
        errors = self.validate()
        if errors:
            raise ValueError('\n'.join(errors[:12]))
        if not str(reviewer).strip():
            raise ValueError('请填写标注人编号再保存')
        target = Path(path).resolve()
        if target.name == 'review.json':
            raise ValueError('review.json 是发放原件，请另存为 reviewed_标注人.json')
        if target.exists():
            existing = json.loads(target.read_text(encoding='utf-8'))
            if existing.get('base_snapshot_sha256') != self._base_digest:
                raise ValueError('目标属于另一视频或另一批次，请另存新文件')
            if existing.get('reviewer') not in (None, '', reviewer):
                raise ValueError('不能覆盖另一位标注人的回传文件，请另存新文件')
        target.parent.mkdir(parents=True, exist_ok=True)
        result = copy.deepcopy(self.data)
        result.update(video_path=os.path.relpath(self.video_path, target.parent),
                      tracks_path=os.path.relpath(self.tracks_path, target.parent),
                      reviewer=reviewer, saved_at_utc=now(), completion=self.completion())
        temporary = target.with_name(target.name + '.tmp.' + str(os.getpid()))
        with temporary.open('w', encoding='utf-8', newline='\n') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, target)
        self.data = result
        self.path = target
        return target
