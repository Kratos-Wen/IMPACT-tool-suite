"""Durable JSON replacement and lossless editor recovery (no model dependencies)."""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def atomic_json(path, data, *, backup=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Never rotate a damaged file over the last usable backup.
        if backup and path.is_file():
            with path.open(encoding='utf-8-sig') as stream:
                json.load(stream)
            backup_path = path.with_name(path.name + '.bak')
            backup_tmp = temporary + '.bak'
            try:
                shutil.copyfile(path, backup_tmp)
                with open(backup_tmp, 'r+b') as stream:
                    os.fsync(stream.fileno())
                os.replace(backup_tmp, backup_path)
            finally:
                if os.path.exists(backup_tmp):
                    os.unlink(backup_tmp)
        os.replace(temporary, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def content_digest(snapshot):
    state = dict(snapshot)
    for key in ('current_frame', 'selected_event_id', 'selected_hand_label',
                'target_selected', 'instrument_selected', 'anomaly_selected'):
        state.pop(key, None)
    def portable(value):
        if isinstance(value, dict):
            return {str(k): portable(v) for k,v in value.items()}
        if isinstance(value, (tuple, list)):
            return [portable(v) for v in value]
        return value
    return hashlib.sha256(json.dumps(portable(state), sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def write_recovery(path, envelope, baseline):
    digest = content_digest(envelope['state'])
    if digest == content_digest(baseline):
        return False
    path = Path(path)
    if path.is_file():
        try:
            if json.loads(path.read_text(encoding='utf-8'))['digest'] == digest:
                return False
        except (ValueError, KeyError):
            # Preserve a corrupt primary for diagnosis and retain the .bak file.
            os.replace(path, path.with_name(path.name + '.corrupt'))
    envelope = dict(envelope, schema='impact-editor-recovery-1', digest=digest)
    atomic_json(path, envelope)
    return True


def read_recovery(path, identity):
    path = Path(path)
    errors = []
    for candidate in (path, path.with_name(path.name + '.bak')):
        if not candidate.is_file():
            continue
        try:
            doc = json.loads(candidate.read_text(encoding='utf-8'))
            if doc.get('schema') != 'impact-editor-recovery-1':
                raise ValueError('Unsupported recovery schema')
            if doc['identity'] != identity:
                raise ValueError('Recovery belongs to another video or crop')
            if doc['digest'] != content_digest(doc['state']):
                raise ValueError('Recovery checksum mismatch')
            return doc, candidate
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError('; '.join(errors))
    return None, None
