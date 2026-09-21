"""Fetch the missing VRX runtime assets from a pinned official Humble commit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

COMMIT = 'dc30ed8d17aa1083fd872edad9c77c69896d2b07'
PREFIXES = ('vrx_gz/models/coast_waves/', 'vrx_gz/models/blue_projectile/',
            'vrx_urdf/wamv_description/models/', 'vrx_urdf/wamv_gazebo/models/',
            'vrx_urdf/vrx_gazebo/models/')


def download(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'ARBoids-reproduction'}), timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def git_hash(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    tree = json.loads(download(f'https://api.github.com/repos/osrf/vrx/git/trees/{COMMIT}?recursive=1'))
    if tree.get('truncated') or tree.get('sha') != COMMIT:
        raise RuntimeError('Incomplete or unexpected upstream tree')
    entries = [e for e in tree['tree'] if e['type'] == 'blob' and e['path'].startswith(PREFIXES)]
    if not entries:
        raise RuntimeError('No assets found')

    def fetch(entry):
        path = repo / 'vrx' / entry['path']
        if path.is_file() and git_hash(path.read_bytes()) == entry['sha']:
            return
        data = download(f'https://raw.githubusercontent.com/osrf/vrx/{COMMIT}/{entry["path"]}')
        if git_hash(data) != entry['sha']:
            raise RuntimeError(f'Asset checksum mismatch: {entry["path"]}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f'[ASSET] {entry["path"]} ({len(data)} bytes)', flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, entries))
    # These packages install their models directory even when no static models are needed.
    for package in ('vrx_urdf/wamv_gazebo', 'vrx_urdf/vrx_gazebo'):
        (repo / 'vrx' / package / 'models').mkdir(parents=True, exist_ok=True)
    asset_dir = repo / '.vrx-assets'
    asset_dir.mkdir(exist_ok=True)
    license_entry = next(e for e in tree['tree'] if e['path'] == 'LICENSE')
    license_data = download(f'https://raw.githubusercontent.com/osrf/vrx/{COMMIT}/LICENSE')
    if git_hash(license_data) != license_entry['sha']:
        raise RuntimeError('Upstream license checksum mismatch')
    (asset_dir / 'VRX-LICENSE').write_bytes(license_data)
    (asset_dir / 'manifest.json').write_text(json.dumps({'repository': 'https://github.com/osrf/vrx',
                                                       'commit': COMMIT, 'license': license_entry,
                                                       'files': entries}, indent=2), encoding='utf-8')
    print(f'[DONE] {len(entries)} verified official VRX assets', flush=True)


if __name__ == '__main__':
    main()
