"""Update downloaded review pages in place, preserving task data and resume keys."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from render import render_page


def update(path):
    original = path.read_text(encoding='utf-8')
    match = re.search(r'const\s+DATA\s*=\s*', original)
    if not match:
        raise ValueError('No embedded audio-review manifest: ' + str(path))
    data = json.JSONDecoder().raw_decode(original, match.end())[0]
    if not isinstance(data, dict) or not all(k in data for k in ('plan_hash', 'files', 'items', 'language')):
        raise ValueError('Unsupported manifest; original file preserved: ' + str(path))
    lang = re.search(r'<html\s+lang=["\']([^"\']+)', original)
    result = render_page(data, lang[1] if lang else None)
    if result == original:
        return False
    # Preserve the original bytes before an atomic replacement. No audio is read.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    backup = path.parent / '.impact-recovery' / ('review-' + digest + '.html')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    pending = path.with_name(path.name + '.update.tmp')
    pending.write_text(result, encoding='utf-8')
    pending.replace(path)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='Your audio folder, personal task folder, or review.html')
    args = parser.parse_args()
    pages = [args.path] if args.path.is_file() else sorted(p for p in args.path.rglob('review.html') if '.impact-recovery' not in p.parts)
    if not pages:
        parser.error('No review.html found')
    changed = sum(update(p) for p in pages)
    print(f'Updated {changed} / {len(pages)} pages. Reopen review.html. Audio and review JSON files are unchanged.')
