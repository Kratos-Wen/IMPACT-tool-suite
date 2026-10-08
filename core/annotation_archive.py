"""Preserve a local imported source before any schema conversion."""
import hashlib,os
from pathlib import Path

def archive_source(path):
    src=Path(path)
    payload=src.read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    directory=src.parent/'archived'/digest
    directory.mkdir(parents=True,exist_ok=True)
    dst=directory/src.name
    if dst.exists():
        if dst.read_bytes()!=payload:raise ValueError('Archive collision')
    else:
        # Exclusive creation prevents concurrent imports from overwriting originals.
        try:
            with dst.open('xb') as stream:stream.write(payload)
        except FileExistsError:
            if dst.read_bytes()!=payload:raise ValueError('Archive collision')
    return {'path':str(dst.relative_to(src.parent)),'sha256':digest}
