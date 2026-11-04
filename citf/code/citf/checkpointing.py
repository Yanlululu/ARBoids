"""Checkpoint-owned log commits; unfinished attempts remain recoverable evidence."""
import base64
import hashlib
import json
import os
from pathlib import Path

from .contracts import canonical_hash


def atomic_bytes(path, data):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def atomic_json(value, path):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8'))


class CheckpointJournal:
    """The checkpoint commits byte prefixes of the existing JSONL files.

    A failed update cannot commit newer logs by saving the old model again.
    Recovery durably retains every uncommitted byte before restoring prefixes;
    a retry of recovery is idempotent, including after a partial JSONL write.
    """
    def __init__(self, directory, names):
        self.directory = Path(directory)
        self.names = tuple(names)
        self.recovery_path = self.directory / 'uncommitted_records.jsonl'

    def rows(self, name):
        path = self.directory / name
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []

    def append(self, name, rows):
        if name not in self.names:
            raise ValueError('Undeclared journal stream')
        data = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows).encode('utf-8')
        with (self.directory / name).open('ab') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def snapshot(self):
        result = {}
        for name in self.names:
            path = self.directory / name
            raw = path.read_bytes() if path.exists() else b''
            result[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return result

    def recover(self, committed, update):
        if committed is None or set(committed) != set(self.names):
            raise ValueError('Checkpoint lacks journal commit boundaries; retain the legacy run separately')
        pending = []
        # Check every committed prefix before changing any working log.
        for name in self.names:
            path = self.directory / name
            raw = path.read_bytes() if path.exists() else b''
            mark = committed[name]
            size = mark['bytes']
            if size < 0 or len(raw) < size or hashlib.sha256(raw[:size]).hexdigest() != mark['sha256']:
                raise ValueError('Committed checkpoint log was changed: ' + name)
            if len(raw) > size:
                tail = raw[size:]
                key = canonical_hash(dict(stream=name, prefix=mark, tail=hashlib.sha256(tail).hexdigest()))
                rows, partial = [], 0
                for line in tail.splitlines():
                    try:
                        rows.append(json.loads(line))
                    except (ValueError, UnicodeDecodeError):
                        partial += 1
                episodes = [row for row in rows if isinstance(row, dict)] if name == 'episodes.jsonl' else []
                record = dict(recovery_id=key, stream=name, committed_update=update,
                    committed_bytes=size, uncommitted_bytes=len(tail),
                    raw_base64=base64.b64encode(tail).decode('ascii'), records=rows,
                    partial_records=partial, recorded_episode_attempts=len(episodes),
                    recorded_environment_steps=sum(row.get('steps', 0) for row in episodes))
                pending.append((path, raw[:size], record))
        archive = self.rows(self.recovery_path.name)
        known = {row['recovery_id'] for row in archive}
        additions = [row for _, _, row in pending if row['recovery_id'] not in known]
        if additions:
            archive.extend(additions)
            atomic_bytes(self.recovery_path, ''.join(json.dumps(row) + '\n' for row in archive).encode('utf-8'))
        # The durable recovery journal now owns the tails, even if interrupted here.
        for path, prefix, _ in pending:
            atomic_bytes(path, prefix)
        return dict(restored_update=update, recovered_streams=[row['stream'] for _, _, row in pending],
            recorded_uncommitted_episode_attempts=sum(row['recorded_episode_attempts'] for row in archive),
            recorded_uncommitted_environment_steps=sum(row['recorded_environment_steps'] for row in archive),
            evidence=self.recovery_path.name if archive else None)
