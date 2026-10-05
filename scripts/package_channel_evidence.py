"""Create a minimal experiment transfer bundle with a content manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--channel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = sorted(p for p in (root / 'train').rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.yaml')
                   and 'experiments' not in p.relative_to(root / 'train').parts)
    files += sorted((root / 'vrx').glob('*.py'))
    files += [root / 'docs/channel-evidence-protocol.md', Path(__file__).resolve()]
    inputs = [(args.reference, 'inputs/arboids-reference.pth'),
              (args.channel, 'inputs/channel-frozen.pth')]
    checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(protocol='channel-evidence-v1',
                    files={p.relative_to(root).as_posix(): checksum(p) for p in files},
                    inputs={name: checksum(p) for p, name in inputs})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'source-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = args.output / 'deployment.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in files:
            bundle.add(path, arcname=path.relative_to(root).as_posix())
        for path, name in inputs:
            bundle.add(path, arcname=name)
        bundle.add(manifest_path, arcname='source-manifest.json')
    print(json.dumps(dict(source_files=len(files), archive_bytes=archive.stat().st_size,
                          manifest_sha256=checksum(manifest_path))))


if __name__ == '__main__':
    main()
