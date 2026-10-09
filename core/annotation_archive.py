"""Preserve a local imported source before any schema conversion."""
import hashlib,os
from pathlib import Path

def archive_source(path):
    src=Path(path)
    payload=src.read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    # Imported originals belong in the application cache, never in reviewer tasks.
    root=Path(os.environ.get('IMPACT_SOURCE_ARCHIVE', str(Path.home()/'.impact-tool-suite'/'import_sources')))
    directory=root/digest
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
    return {'sha256':digest,'storage':'application_import_cache','filename':src.name}
