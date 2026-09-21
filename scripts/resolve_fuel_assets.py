"""Resolve recursive Fuel references locally without changing upstream assets."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

MODEL_URL = re.compile(r'(https://fuel\.(?:gazebosim\.org|ignitionrobotics\.org)/1\.0/[^/]+/models/[^/]+)(/.*)?')


def resolve(repo):
    assets = repo.resolve() / '.vrx-assets'
    manifest = json.loads((assets/'fuel-manifest.json').read_text(encoding='utf-8'))
    entries = {entry['url']: entry for entry in manifest}
    destinations = {url: assets/'resolved'/entry['directory'] for url, entry in entries.items()}
    for directory in destinations.values():
        directory.mkdir(parents=True, exist_ok=True)
    replacements = 0
    for url, entry in entries.items():
        source = assets/'fuel'/entry['directory']
        destination = destinations[url]
        for child in source.iterdir():
            target = destination/child.name
            if child.suffix != '.sdf':
                if target.is_symlink():
                    if target.readlink() != child:
                        target.unlink()
                    else:
                        continue
                if target.exists():
                    raise RuntimeError(f'Expected an asset symlink: {target}')
                target.symlink_to(child, target_is_directory=child.is_dir())
                continue
            tree = ET.parse(child)
            for element in tree.iter():
                if not element.text or not (match := MODEL_URL.fullmatch(element.text.strip())):
                    continue
                dependency = match[1].replace('fuel.ignitionrobotics.org', 'fuel.gazebosim.org')
                if dependency not in entries:
                    raise FileNotFoundError(f'Rerun fetch_fuel_assets.py for {dependency}')
                suffix = match[2] or ''
                if '/files/' in suffix:
                    relative = suffix.split('/files/', 1)[1]
                    raw = assets/'fuel'/entries[dependency]['directory']
                    resolved = (raw/relative).resolve()
                    if not resolved.is_relative_to(raw.resolve()) or not resolved.is_file():
                        raise FileNotFoundError(f'Missing model resource: {element.text}')
                else:
                    resolved = destinations[dependency]
                element.text = str(resolved)
                replacements += 1
            tree.write(target, encoding='utf-8', xml_declaration=True)
    print(f'[RESOLVED] {len(entries)} models; {replacements} nested Fuel references', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    resolve(parser.parse_args().repo)
