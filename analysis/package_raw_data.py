"""Losslessly package the original audit inputs for Git LFS; standard library only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'TBMData/TBMData'
DEST = ROOT / 'data/raw'
CHUNK = 1024 * 1024

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''): h.update(block)
    return h.hexdigest()

def package(task):
    name, paths = task
    target = DEST / name
    entries = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 6
            info.external_attr = 0o100644 << 16
            h = hashlib.sha256(); count = 0
            with path.open('rb') as source, z.open(info, 'w', force_zip64=True) as output:
                for block in iter(lambda: source.read(CHUNK), b''):
                    h.update(block); output.write(block); count += len(block)
            entries.append(dict(path=relative, bytes=count, sha256=h.hexdigest()))
    result = dict(file=target.relative_to(ROOT).as_posix(), bytes=target.stat().st_size,
                  sha256=sha256_file(target), entries=entries)
    print('PACKED', name, len(entries), 'files', target.stat().st_size, 'bytes', flush=True)
    return result

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    tasks = []; discovered = []
    for d in sorted(SOURCE.iterdir()):
        if not d.is_dir(): continue
        paths = sorted(p for p in d.rglob('tbm_data*.csv') if 'common_using' not in p.parts and 'codes' not in p.parts)
        if paths:
            tasks.append((f'source-{len(tasks)+1:02d}.zip', paths)); discovered.extend(paths)
    with (ROOT/'analysis/outputs/file_audit.csv').open(encoding='utf-8-sig', newline='') as f:
        audited = {row['file'].replace('\\', '/') for row in csv.DictReader(f)}
    assert {p.relative_to(SOURCE).as_posix() for p in discovered} == audited
    workbooks = sorted(SOURCE.glob('*.xlsx'))
    if workbooks: tasks.append(('workbook.zip', workbooks)); discovered.extend(workbooks)
    with ThreadPoolExecutor(max_workers=4) as pool: archives = list(pool.map(package, tasks))
    manifest = dict(format_version=1, source_date='2026-10-28', compression='ZIP DEFLATE, lossless',
                    source_files=len(discovered), source_bytes=sum(p.stat().st_size for p in discovered),
                    archive_bytes=sum(a['bytes'] for a in archives), archives=archives,
                    policy='All 1531 audited source CSVs and the original workbook, including quarantined and incomplete inputs; no environments or derived data.')
    (DEST/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('COMPLETE',manifest['source_files'],manifest['source_bytes'],manifest['archive_bytes'],flush=True)

if __name__ == '__main__': main()
