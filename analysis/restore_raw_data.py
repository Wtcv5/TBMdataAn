"""Verify and restore LFS raw archives. Existing matching files are kept; differing files are refused."""
from pathlib import Path
import argparse, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''): h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify every archive and member without extracting.')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'data/raw/manifest.json').read_text(encoding='utf-8'))
    checked = 0; total = 0
    for archive in manifest['archives']:
        path = ROOT / archive['file']
        if path.stat().st_size != archive['bytes'] or digest_file(path) != archive['sha256']:
            raise ValueError(f'Archive mismatch: {path}. Run git lfs pull to download full archives.')
        with zipfile.ZipFile(path) as z:
            assert set(z.namelist()) == {entry['path'] for entry in archive['entries']}
            for entry in archive['entries']:
                target = (ROOT / entry['path']).resolve()
                if not target.is_relative_to((ROOT/'TBMData/TBMData').resolve()):
                    raise ValueError(f'Unsafe member path: {entry["path"]}')
                if not args.verify_only and target.exists():
                    if target.stat().st_size != entry['bytes'] or digest_file(target) != entry['sha256']:
                        raise FileExistsError(f'Refusing to overwrite different existing file: {target}')
                    checked += 1; total += entry['bytes']; continue
                h = hashlib.sha256(); count = 0
                if not args.verify_only:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Exclusive creation avoids overwriting files that appear during verification.
                    destination = target.open('xb')
                else: destination = None
                try:
                    with z.open(entry['path']) as f:
                        for block in iter(lambda: f.read(CHUNK), b''):
                            h.update(block); count += len(block)
                            if destination: destination.write(block)
                finally:
                    if destination: destination.close()
                if count != entry['bytes'] or h.hexdigest() != entry['sha256']:
                    raise ValueError(f'Member mismatch: {entry["path"]}')
                checked += 1; total += count
        print('VERIFIED',archive['file'],flush=True)
    assert checked == manifest['source_files'] and total == manifest['source_bytes']
    result = dict(status='passed',files=checked,source_bytes=total,archive_bytes=manifest['archive_bytes'],
                  verified_content='SHA-256 of all archives and every decompressed original file',
                  mode='verify_only' if args.verify_only else 'restore')
    (ROOT/'data/raw/verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result,flush=True)

if __name__ == '__main__': main()
