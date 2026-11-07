"""Launch a registered entry with its complete, immutable CITF source bundle.

The campaign copy reads current_source.json once. The copy inside a bundle
always uses that bundle. Existing processes and historical snapshots are not
redirected. This is a one-shot launcher, not a queue or background service.
"""
import argparse
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys


def bind_release(release=None):
    if release is None:
        own_root = Path(__file__).resolve().parents[2]
        if (own_root/'citf/release/source_bundle.json').is_file():
            release = own_root
        else:
            pointer = Path(__file__).resolve().with_name('current_source.json')
            release = json.loads(pointer.read_text(encoding='utf-8'))['release_root']
    root = Path(release).resolve()
    manifest = json.loads((root/'citf/release/source_bundle.json').read_text(encoding='utf-8'))
    if manifest['release_root'] != str(root):
        raise ValueError('The frozen bundle location differs from its registered source root.')
    code, scripts, vrx = root/'citf/code', root/'citf/tools', root/'vrx'
    loaded = sys.modules.get('unified')
    if loaded is not None and Path(loaded.__file__).resolve().parent != code/'unified':
        raise RuntimeError('An unrelated unified package was already loaded; launch a fresh process.')
    os.environ.update(CITF_RELEASE_ROOT=str(root), CITF_RELEASE_ID=manifest['release_id'],
                      CITF_CODE_ROOT=str(code), PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    # Precedence also applies to study/VRX subprocesses. Preserve ROS paths.
    prefixes = [str(code), str(scripts), str(vrx)]
    old_pythonpath = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    os.environ['PYTHONPATH'] = os.pathsep.join(prefixes+[p for p in old_pythonpath if p and p not in prefixes])
    sys.path[:] = prefixes+[p for p in sys.path if p not in prefixes]
    return root, manifest


def entry_definition(entry, manifest):
    aliases = {'train': 'unified.train', 'baseline': 'unified.train_baseline',
               'evaluate': 'unified.evaluate', 'study': 'unified.run_study',
               'alternating': 'unified.train_alternating', 'vrx': 'vrx/run_batch.py'}
    name = aliases.get(entry, entry)
    if name not in manifest['entries']:
        raise ValueError(f'Entry {name!r} is not in this frozen bundle; freeze its paired source before launch.')
    return name, manifest['entries'][name]


def load_entry(entry, root, manifest):
    name, relative = entry_definition(entry, manifest)
    target = root/relative
    module_name = name if name.startswith('unified.') else target.stem
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != target:
        raise RuntimeError(f'{name} loaded outside the selected bundle: {module.__file__}')
    return module, target


def export_entry(entry, namespace):
    """Keep existing import-style tool entry points on the complete release."""
    root, manifest = bind_release()
    _, relative = entry_definition(entry, manifest)
    target = root/relative
    name = '_citf_release_'+manifest['release_id'].replace('-', '_')+'_'+target.stem
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    namespace.update({key: value for key, value in vars(module).items() if not key.startswith('__')})
    namespace['__file__'] = str(target)


def activate_release(root, manifest, campaign):
    campaign = Path(campaign).resolve()
    if root.parent != campaign:
        raise ValueError('Publish the bundle only in its registered campaign directory.')
    aliases = {name: name for name, path in manifest['entries'].items() if path.startswith('citf/tools/')}
    aliases['run_vrx_batch.py'] = 'vrx/run_batch.py'
    protected = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            argv = [v.decode() for v in (process/'cmdline').read_bytes().split(b'\0') if v]
            if any(str(campaign/name) in argv for name in (*aliases, 'launch_current.py')):
                raise RuntimeError(f'An active process uses a campaign entry being replaced: {process.name}')
            if any(str(campaign) in value for value in argv):
                protected.append(dict(pid=int(process.name), cwd=str((process/'cwd').resolve())))
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
    def atomic(path, content):
        temporary = path.with_suffix('.publishing')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)
    launcher = campaign/'launch_current.py'
    atomic(launcher, (root/'citf/tools/launch_frozen.py').read_text(encoding='utf-8'))
    pointer = dict(status='active', release_id=manifest['release_id'], release_root=str(root),
                   launcher=str(launcher), code=str(root/'citf/code'), tools=str(root/'citf/tools'),
                   vrx=str(root/'vrx'), aliases=aliases, existing_processes_preserved=protected)
    atomic(campaign/'current_source.json', json.dumps(pointer, ensure_ascii=False, indent=2)+'\n')
    for filename, entry in aliases.items():
        proxy = ('"""Stable entry for the active complete CITF release."""\n'
                 'import runpy, sys\n'
                 f'_launcher = runpy.run_path({str(launcher)!r})\n'
                 'if __name__ == "__main__":\n'
                 f'    _launcher["main"]([{entry!r}, *sys.argv[1:]])\n'
                 'else:\n'
                 f'    _launcher["export_entry"]({entry!r}, globals())\n')
        atomic(campaign/filename, proxy)
    return pointer


def inspect_release(root, manifest, entry=None):
    modules = {}
    for name in ('unified', 'unified.protocol', 'unified.train', 'unified.learner',
                 'unified.networks', 'unified.environment', 'unified.baselines',
                 'train_shared_team_interaction'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f'{name} came from a different source: {path}')
        modules[name] = str(path)
    protocol = sys.modules['unified.protocol']
    if ('current_source' not in inspect.signature(protocol.validate_checkpoint).parameters
            or not hasattr(protocol, 'require_standard_controller')
            or not hasattr(sys.modules['unified.train'], 'fork_arguments')):
        raise RuntimeError('The selected source is missing the integrated handoff repairs.')
    if entry:
        module, target = load_entry(entry, root, manifest)
        modules[entry] = str(target)
        if str(target).startswith(str(root/'vrx')):
            runtime = importlib.import_module('unified_vrx')
            modules['unified_vrx'] = str(Path(runtime.__file__).resolve())
    return dict(event='frozen_source_loaded', release_id=manifest['release_id'],
                release_root=str(root), entry=entry, pid=os.getpid(),
                python=sys.executable, modules=modules)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', help='Pin an explicit complete release instead of the active pointer.')
    parser.add_argument('--inspect', action='store_true', help='Load and report the actual source without starting a research job.')
    parser.add_argument('--activate', help='Publish this complete release as the campaign default without altering running snapshots.')
    parser.add_argument('entry', nargs='?', help='Registered module, tool filename, or vrx entry.')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.entry is None and not args.inspect and not args.activate:
        parser.error('Specify an entry, or use --inspect to read the active release.')
    root, manifest = bind_release(args.release)
    receipt = inspect_release(root, manifest, args.entry)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if args.activate:
        if args.entry or args.arguments:
            parser.error('Activation publishes source only; launch research jobs separately.')
        print(json.dumps(activate_release(root, manifest, args.activate), ensure_ascii=False), flush=True)
        return
    if args.inspect:
        return
    name, relative = entry_definition(args.entry, manifest)
    target = root/relative
    sys.argv = [str(target), *args.arguments]
    if name.startswith('unified.'):
        importlib.import_module(name).main()
    else:
        # Candidate tools supply their specific model/learner in __main__.
        runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
