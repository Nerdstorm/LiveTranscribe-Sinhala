#!/usr/bin/env python3
r"""Downloads FLEURS transcripts and audio from Hugging Face, checks them, and extracts the audio.

    fetch_fleurs.py --out <folder> [--languages en_us,cmn_hans_cn,es_419,fr_fr,de_de] \
        [--splits train] [--extract]

Fetches data/<code>/<split>.tsv and data/<code>/audio/<split>.tar.gz of google/fleurs
(https://huggingface.co/datasets/google/fleurs, CC BY 4.0), at the revision the replay set was
made from, into <folder>/data/<code>/. Each file is checked against the size and hash the Hub
lists: SHA-256 for the archives, the git object id for the transcripts. A file already there that
passes isn't fetched again, and a download that broke off carries on from where it stopped. With
--extract, each split's recordings go to <folder>/data/<code>/audio/<split>/, skipping those
already there.

The default languages are the replay set's. For the English check in docs/training.md, use
--languages en_us --splits test. Stdlib only.
"""
import argparse
import json
import os
import sys
import tarfile
import urllib.request

from fetching import TIMEOUT, download, extract, verified
from make_replay import FLEURS_CODES, parse_sources

REPOSITORY = "google/fleurs"
# The revision the replay set was made from: the Hub's main on 2026-09-25.
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FILES = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"


def listing(code):
    """The Hub's entries for data/<code>: path, size and oid, and lfs {oid, size} for large files."""
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/data/{code}?recursive=true"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return [entry for entry in json.load(response) if entry.get("type") == "file"]


def wanted(entries, code, splits):
    """(path, size, hash kind, hash) of each split's transcript and archive. Stops if the Hub
    doesn't have one."""
    paths = {f"data/{code}/{split}.tsv" for split in splits} | {f"data/{code}/audio/{split}.tar.gz" for split in splits}
    files = []
    for entry in entries:
        if entry["path"] not in paths:
            continue
        lfs = entry.get("lfs")
        if lfs:
            files.append((entry["path"], lfs["size"], "sha256", lfs["oid"]))
        else:
            files.append((entry["path"], entry["size"], "git", entry["oid"]))
    missing = paths - {path for path, *_ in files}
    if missing:
        raise ValueError(f"FLEURS has no {', '.join(sorted(missing))}")
    return sorted(files)


def fetch(out, codes, splits, extract_audio, list_files=listing, files_url=FILES):
    for code in codes:
        for path, size, kind, expected in wanted(list_files(code), code, splits):
            target = os.path.join(out, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if verified(target, size, kind, expected):
                print(f"have {path}", flush=True)
                continue
            print(f"fetching {path} ({size / 1e6:.0f} MB)", flush=True)
            download([f"{files_url}/{path}"], target, size, kind, expected)
        if extract_audio:
            audio = os.path.join(out, "data", code, "audio")
            for split in splits:
                written = extract(os.path.join(audio, f"{split}.tar.gz"), audio, split)
                print(f"extracted {written} recordings to {os.path.join(audio, split)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="folder for FLEURS's data/ tree")
    parser.add_argument("--languages", default=",".join(FLEURS_CODES), help="FLEURS codes, separated by commas (default: the replay set's)")
    parser.add_argument("--splits", default="train", help="train, dev or test, separated by commas (default train)")
    parser.add_argument("--extract", action="store_true", help="also extract the recordings")
    args = parser.parse_args()

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    try:
        if not splits or set(splits) - {"train", "dev", "test"}:
            raise ValueError(f"unknown splits {args.splits!r}: use train, dev or test")
        fetch(args.out, parse_sources(args.languages, FLEURS_CODES), splits, args.extract)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        sys.exit(f"fetch_fleurs: {error}")


if __name__ == "__main__":
    main()
