"""Checked downloads and safe extraction, for fetch_fleurs.py and fetch_librispeech.py.

A module the fetch scripts share, not a script. Stdlib only.
"""
import hashlib
import http.client
import os
import shutil
import sys
import tarfile
import urllib.request

ATTEMPTS = 3
TIMEOUT = 60


def digest(path, kind):
    """The file's SHA-256 or MD5 ("sha256", "md5"), or its git object id ("git": the SHA-1 of a
    "blob <size>" header and the bytes)."""
    if kind == "git":
        hasher = hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
    elif kind in {"sha256", "md5"}:
        hasher = hashlib.new(kind)
    else:
        raise ValueError(f"unknown hash {kind!r}")
    with open(path, "rb") as data:
        for block in iter(lambda: data.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verified(path, size, kind, expected):
    return os.path.isfile(path) and os.path.getsize(path) == size and digest(path, kind) == expected


def fetch_rest(url, partial, size):
    """Writes the file at url into partial, asking only for the bytes partial doesn't have yet. A
    server that sends the whole file instead (no 206 for the range asked for) starts it over."""
    have = os.path.getsize(partial) if os.path.isfile(partial) else 0
    if have and have >= size:
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        resumed = (have and getattr(response, "status", None) == 206
                   and response.headers.get("Content-Range", "").startswith(f"bytes {have}-"))
        with open(partial, "ab" if resumed else "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)


def download(urls, target, size, kind, expected, attempts=ATTEMPTS):
    """Fetches a file into target through target.part, trying each URL (the mirrors) in turn, and
    keeps it only if it has the size and hash published for it. An attempt that breaks off leaves
    target.part, and the next attempt, or the next run, carries on from it; a finished file that
    doesn't match is thrown away and fetched again from the start."""
    partial = target + ".part"
    problem = "no attempt made"
    for attempt in range(1, attempts + 1):
        for url in urls:
            try:
                fetch_rest(url, partial, size)
                if verified(partial, size, kind, expected):
                    os.replace(partial, target)
                    return
                os.remove(partial)
                problem = "the file doesn't match the size and hash published for it"
            except (OSError, http.client.HTTPException) as error:
                problem = str(error) or type(error).__name__
            print(f"attempt {attempt} of {attempts} at {url} failed: {problem}", file=sys.stderr)
    kept = f"; run again to carry on from the {os.path.getsize(partial):,} bytes fetched" if os.path.isfile(partial) else ""
    raise RuntimeError(f"couldn't fetch {os.path.basename(target)}: {problem}{kept}")


def extract(archive, destination, top):
    """Writes the archive's files to destination/<top>/…, skipping those already there with the
    right size, and returns how many it wrote. Stops at an entry outside <top>/ and at anything
    that isn't a plain file or folder, such as a link."""
    written = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            parts = member.name.rstrip("/").split("/")
            inside = parts[0] == top and all(part not in {"", ".", ".."} for part in parts)
            if not inside or not (member.isfile() or member.isdir()):
                raise ValueError(f"{archive}: unexpected entry {member.name!r}")
            target = os.path.join(destination, *parts)
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            if os.path.isfile(target) and os.path.getsize(target) == member.size:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tar.extractfile(member) as source, open(target + ".part", "wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
            os.replace(target + ".part", target)
            written += 1
    return written
