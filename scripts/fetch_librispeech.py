#!/usr/bin/env python3
"""Downloads LibriSpeech's train-clean-100, checks it against OpenSLR's MD5, and extracts it.

    fetch_librispeech.py --out <folder> [--extract]

Fetches train-clean-100.tar.gz of LibriSpeech (https://www.openslr.org/12/, CC BY 4.0, 6.4 GB)
into <folder>, trying the mirrors in turn, and checks its size and the MD5 OpenSLR publishes. A
file already there that passes isn't fetched again, and a download that broke off carries on from
where it stopped. With --extract, the recordings (FLAC, 16 kHz) and transcripts go to
<folder>/LibriSpeech/train-clean-100/<speaker>/<chapter>/, skipping those already there. Stdlib
only.
"""
import argparse
import os
import sys
import tarfile

from fetching import download, extract, verified
from make_replay import LIBRISPEECH_SUBSET

ARCHIVE = f"{LIBRISPEECH_SUBSET}.tar.gz"
SIZE = 6_387_309_499
# From https://www.openslr.org/resources/12/md5sum.txt
MD5 = "2a93770f6d5c6c964bc36631d331a522"
# OpenSLR's Chinese mirror sends downloads to the Hugging Face copy, which is the fastest, and the
# MD5 check holds every copy to OpenSLR's file.
MIRRORS = (
    "https://huggingface.co/datasets/k2-fsa/LibriSpeech/resolve/2dd4206a992cd7a475e1ce2e82b708b014319152",
    "https://www.openslr.org/resources/12",
    "https://openslr.elda.org/resources/12",
)


def fetch(out, extract_files, mirrors=MIRRORS, size=SIZE, md5=MD5):
    os.makedirs(out, exist_ok=True)
    archive = os.path.join(out, ARCHIVE)
    if verified(archive, size, "md5", md5):
        print(f"have {ARCHIVE}", flush=True)
    else:
        print(f"fetching {ARCHIVE} ({size / 1e9:.1f} GB)", flush=True)
        download([f"{mirror}/{ARCHIVE}" for mirror in mirrors], archive, size, "md5", md5)
    if extract_files:
        written = extract(archive, out, "LibriSpeech")
        print(f"extracted {written} files to {os.path.join(out, 'LibriSpeech', LIBRISPEECH_SUBSET)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="folder for the archive and, with --extract, LibriSpeech/")
    parser.add_argument("--extract", action="store_true", help="also extract the recordings and transcripts")
    args = parser.parse_args()
    try:
        fetch(args.out, args.extract)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        sys.exit(f"fetch_librispeech: {error}")


if __name__ == "__main__":
    main()
