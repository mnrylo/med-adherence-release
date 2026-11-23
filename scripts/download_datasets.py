"""Download, verify and extract the third-party datasets into data/third_party/.

    python scripts/download_datasets.py                 # both datasets
    python scripts/download_datasets.py wisdm
    python scripts/download_datasets.py ingesture --archive ~/Downloads/fdxst56tcj-5.zip

* InGesture v5 (Mendeley Data, doi:10.17632/fdxst56tcj.5, CC BY 4.0). Mendeley Data does not
  always serve the archive to scripts; if the download fails, download "fdxst56tcj-5.zip" from
  https://data.mendeley.com/datasets/fdxst56tcj/5 ("Download All") and pass it with --archive.
* WISDM 2019 (UCI Machine Learning Repository, doi:10.24432/C5HK59, CC BY 4.0). The UCI download
  holds an inner archive, wisdm-dataset.zip, which is the one verified and extracted.

Every archive is checked against its SHA-256 before extraction; extracted files are not modified.
The target paths match conf/paths.json. Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = json.loads((ROOT / "conf" / "paths.json").read_text())

DATASETS = {
    "ingesture": {
        "urls": ["https://data.mendeley.com/public-api/zip/fdxst56tcj/download/5",
                 "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fdxst56tcj-5.zip"],
        "page": "https://data.mendeley.com/datasets/fdxst56tcj/5",
        "archive": "fdxst56tcj-5.zip",
        "inner": None,
        "sha256": "0564f8debac98c284796681044ee792c58c8ed599ce71e3113fe39916ac25325",
        "target": PATHS["ingesture"],
        "check": "ingesture_metadata.xlsx",
    },
    "wisdm": {
        "urls": ["https://archive.ics.uci.edu/static/public/507/"
                 "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"],
        "page": "https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset",
        "archive": "wisdm-dataset.zip",
        "inner": "wisdm-dataset.zip",
        "sha256": "1cf32e8d3a0ecc101bd299fba22545273dcb7de1c9565e0b7844abdfbc6b1695",
        "target": PATHS["wisdm"],
        "check": "raw/watch/accel",
    },
}


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream a URL to dest (written to a temporary file first)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} MiB", end="", file=sys.stderr)
    print(file=sys.stderr)
    tmp.replace(dest)


def extract_inner(outer: Path, name: str, dest: Path) -> None:
    """Copy the inner archive ``name`` of an outer zip to dest."""
    with zipfile.ZipFile(outer) as z:
        member = next(m for m in z.namelist() if Path(m).name == name)
        with z.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)


def fetch(name: str, archive: Path | None, keep: bool) -> Path:
    """Verified archive of a dataset under data/third_party/<name>/ (downloaded unless given)."""
    spec = DATASETS[name]
    target = ROOT / spec["target"]
    store = target.parent
    store.mkdir(parents=True, exist_ok=True)
    dest = store / spec["archive"]
    if archive is not None:
        archive = archive.expanduser().resolve()
        if spec["inner"] and archive.name != spec["archive"]:
            extract_inner(archive, spec["inner"], dest)
        elif archive != dest:
            shutil.copy2(archive, dest)
    elif not dest.exists():
        last = None
        for url in spec["urls"]:
            print(f"{name}: downloading {url}", file=sys.stderr)
            try:
                raw = dest.with_suffix(".download")
                download(url, raw)
                if spec["inner"]:
                    extract_inner(raw, spec["inner"], dest)
                    if not keep:
                        raw.unlink()
                else:
                    raw.replace(dest)
                break
            except Exception as e:
                last = e
                print(f"{name}: {url} failed ({e})", file=sys.stderr)
        else:
            raise SystemExit(f"{name}: automatic download failed ({last}).\n"
                             f"Download {spec['archive']} from {spec['page']} and run\n"
                             f"  python scripts/download_datasets.py {name} --archive <path to the zip>")
    got = sha256(dest)
    if got != spec["sha256"]:
        raise SystemExit(f"{name}: SHA-256 mismatch for {dest}\n  expected {spec['sha256']}\n  got      {got}")
    print(f"{name}: {dest.name} verified (SHA-256 {got[:12]}...)", file=sys.stderr)
    return dest


def extract(name: str, dest: Path) -> Path:
    """Extract a verified archive next to it; returns the dataset directory."""
    spec = DATASETS[name]
    target = ROOT / spec["target"]
    if (target / spec["check"]).exists():
        print(f"{name}: already extracted at {target}", file=sys.stderr)
        return target
    with zipfile.ZipFile(dest) as z:
        for m in z.infolist():
            p = (dest.parent / m.filename).resolve()
            if not str(p).startswith(str(dest.parent.resolve())):
                raise SystemExit(f"{name}: unsafe path in archive: {m.filename}")
        z.extractall(dest.parent)
    if not (target / spec["check"]).exists():
        raise SystemExit(f"{name}: {target / spec['check']} missing after extraction")
    print(f"{name}: extracted to {target}", file=sys.stderr)
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", metavar="DATASET", help=f"any of {', '.join(DATASETS)} (default: all)")
    ap.add_argument("--archive", type=Path, help="use a manually downloaded archive (one dataset only)")
    ap.add_argument("--keep-download", action="store_true", help="keep the outer UCI archive")
    a = ap.parse_args(argv)
    names = a.datasets or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        ap.error(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    if a.archive is not None and len(names) != 1:
        ap.error("--archive needs exactly one dataset name")
    for n in names:
        extract(n, fetch(n, a.archive, a.keep_download))
    return 0


if __name__ == "__main__":
    sys.exit(main())
