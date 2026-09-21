"""Cache the Fuel models actually referenced by the ARBoids evaluation worlds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

from download_ranges import download

MODEL_URL = re.compile(r'(https://fuel\.(?:gazebosim\.org|ignitionrobotics\.org)/1\.0/[^/]+/models/[^/]+)(?:/.*)?')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    cache = repo / '.vrx-assets/fuel'
    cache.mkdir(parents=True, exist_ok=True)
    urls = set()
    for world in ('sydney_regatta_original.sdf', 'sydney_regatta_original1.sdf'):
        tree = ET.parse(repo / 'vrx/vrx_gz/worlds' / world)
        urls.update(e.text.strip() for e in tree.findall('.//include/uri')
                    if e.text and e.text.strip().startswith('https://fuel.gazebosim.org/'))
    def fetch(url):
        with urlopen(url, timeout=30) as response:
            metadata = json.load(response)
        name, version = metadata['name'], str(metadata['version'])
        destination = cache / 'fuel.gazebosim.org' / metadata['owner'].lower() / 'models' / name.lower() / version
        if not (destination/'model.config').is_file():
            archive_path = repo / '.vrx-assets/downloads' / f'{name}-v{version}.zip'
            download(f'{url}/{version}/{name}.zip', archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError(f'Corrupt Fuel archive: {name}')
                names = archive.namelist()
                config_path = next(n for n in names if PurePosixPath(n).name == 'model.config')
                prefix = PurePosixPath(config_path).parent
                for member in archive.infolist():
                    path = PurePosixPath(member.filename)
                    if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(member.external_attr >> 16):
                        raise RuntimeError('Unsafe archive member')
                    relative = path.relative_to(prefix)
                    target = destination / relative
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open('wb') as output:
                            shutil.copyfileobj(source, output)
            archive_path.unlink()
        description = ET.parse(destination/'model.config')
        for sdf in description.findall('.//sdf'):
            if not (destination / sdf.text.strip()).is_file():
                raise RuntimeError(f'Missing SDF in Fuel model: {url}')
        print(f'[FUEL] {name}, version {version}', flush=True)
        dependencies = set()
        for sdf in destination.glob('*.sdf'):
            for element in ET.parse(sdf).iter():
                if element.text and (match := MODEL_URL.fullmatch(element.text.strip())):
                    dependencies.add(match[1].replace('fuel.ignitionrobotics.org', 'fuel.gazebosim.org'))
        return {'url': url, 'version': version, 'directory': str(destination.relative_to(cache)),
                'license': metadata.get('license_name'), 'license_url': metadata.get('license_url'),
                'dependencies': sorted(dependencies)}

    fetched = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        while pending := sorted(urls - fetched.keys()):
            for entry in pool.map(fetch, pending):
                fetched[entry['url']] = entry
                urls.update(entry['dependencies'])
    manifest = sorted(fetched.values(), key=lambda entry: entry['url'])
    (repo / '.vrx-assets/fuel-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[DONE] {len(manifest)} Fuel models cached and checked', flush=True)


if __name__ == '__main__':
    main()
