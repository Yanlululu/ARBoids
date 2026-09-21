"""Download public assets with bounded retries and resumable byte ranges."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from urllib.request import Request, urlopen


def download(url, destination, workers=6):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(url, headers={'Range': 'bytes=0-0'}), timeout=30) as response:
        if response.status != 206:
            raise RuntimeError('Source does not support ranged downloads')
        total = int(response.headers['Content-Range'].split('/')[-1])
        response.read()
    start = destination.stat().st_size if destination.exists() else 0
    if start > total:
        raise RuntimeError('Existing download exceeds source size')
    parts = destination.with_name(destination.name + '.parts')
    parts.mkdir(exist_ok=True)
    step = 8 * 1024 * 1024
    bounds = [(n, min(n+step-1, total-1)) for n in range(start, total, step)]

    def fetch(pair):
        first, last = pair
        part = parts / f'{first}-{last}'
        for attempt in range(10):
            count = part.stat().st_size if part.exists() else 0
            if count == last-first+1:
                return
            if count > last-first+1:
                raise RuntimeError('Oversized partial range')
            offset = first + count
            try:
                request = Request(url, headers={'Range': f'bytes={offset}-{last}'})
                with urlopen(request, timeout=20) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{last}/{total}':
                        raise RuntimeError('Unexpected Content-Range')
                    deadline = time.monotonic() + 150
                    with part.open('ab') as output:
                        while data := response.read(65536):
                            output.write(data)
                            if time.monotonic() > deadline:
                                raise TimeoutError('Slow connection; resume partial range')
                if part.stat().st_size != last-first+1:
                    raise RuntimeError('Incomplete range')
                print(f'[DOWNLOAD] {destination.name}: range ending {last+1}/{total}', flush=True)
                return
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(min(attempt+1, 5))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, bounds))
    if destination.exists() and destination.stat().st_size != start:
        raise RuntimeError('Download changed unexpectedly')
    with destination.open('ab') as output:
        for first, last in bounds:
            with (parts/f'{first}-{last}').open('rb') as source:
                while data := source.read(1024*1024):
                    output.write(data)
    if destination.stat().st_size != total:
        raise RuntimeError('Incorrect final download size')
    for first, last in bounds:
        (parts/f'{first}-{last}').unlink()
    parts.rmdir()
    return destination
