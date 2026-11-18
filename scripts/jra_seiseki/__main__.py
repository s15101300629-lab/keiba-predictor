"""CLI.

    python -m jra_seiseki update                 # 今年の未取得分を取得・変換
    python -m jra_seiseki update --year 2025 2024
    python -m jra_seiseki update --force --limit 3
    python -m jra_seiseki parse path/to/2026-5tokyo4.pdf > out.json
    python -m jra_seiseki rebuild-csv --year 2026
    python -m jra_seiseki pedigree --year 2026      # 出走馬の馬ID・血統を JRA 競走馬情報から取得

Run from the ``scripts`` directory (or with ``PYTHONPATH=scripts``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

from . import PARSER_VERSION
from .fetch import download, list_year
from .parse import parse_pdf
from .store import load_manifest, rebuild_csv, save_manifest, write_day

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "jra-seiseki"
DEFAULT_PEDIGREE = REPO_ROOT / "data" / "jra-pedigree"


def cmd_update(args: argparse.Namespace) -> int:
    out: Path = args.out
    manifest = load_manifest(out)
    years = args.year or [dt.date.today().year]
    touched: set[int] = set()
    failures: list[str] = []
    done = 0
    cache = Path(args.pdf_dir) if args.pdf_dir else Path(tempfile.mkdtemp(prefix="seiseki-"))
    for year in years:
        links = list_year(year)
        print(f"{year}: {len(links)} result PDFs listed", file=sys.stderr)
        for link in sorted(links, key=lambda l: (l.date or "", l.name)):
            entry = manifest.get(link.name)
            if entry and not args.force and entry.get("parser_version") == PARSER_VERSION:
                continue
            if args.limit and done >= args.limit:
                break
            pdf = cache / str(year) / f"{link.name}.pdf"
            try:
                if not pdf.exists():
                    download(link, pdf)
                    time.sleep(args.wait)
                day = parse_pdf(pdf, source_url=link.url)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{link.name}: {e}")
                print(f"  ! {link.name}: {e}", file=sys.stderr)
                continue
            write_day(out, day)
            manifest[link.name] = {
                "url": link.url,
                "date": day.get("date") or link.date,
                "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "races": len(day["races"]),
                "parser_version": PARSER_VERSION,
                "warnings": day["warnings"],
                "unknown_glyphs": day["unknown_glyphs"],
            }
            touched.add(year)
            done += 1
            flag = f" ({len(day['warnings'])} warnings)" if day["warnings"] else ""
            print(f"  + {link.name}: {len(day['races'])} races{flag}", file=sys.stderr)
            if not args.pdf_dir:
                pdf.unlink(missing_ok=True)
    for year in sorted(touched):
        rebuild_csv(out, year)
    save_manifest(out, manifest)
    print(f"converted {done} files; failures: {len(failures)}", file=sys.stderr)
    for f in failures:
        print("  " + f, file=sys.stderr)
    return 1 if failures and not done else 0


def cmd_parse(args: argparse.Namespace) -> int:
    for p in args.pdf:
        day = parse_pdf(p)
        json.dump(day, sys.stdout, ensure_ascii=False, indent=1 if args.pretty else None)
        sys.stdout.write("\n")
        for w in day["warnings"]:
            print("warning: " + w, file=sys.stderr)
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    years = args.year or sorted(int(p.name) for p in (args.out / "json").iterdir() if p.is_dir())
    for y in years:
        rebuild_csv(args.out, y)
    return 0


def cmd_pedigree(args: argparse.Namespace) -> int:
    from .pedigree import run

    converted = sorted((int(p.name) for p in (args.out / "json").iterdir() if p.is_dir()), reverse=True)
    years = args.year or converted
    run(args.pedigree_out, args.out, years, wait=args.wait, workers=args.workers,
        max_minutes=args.max_minutes, limit=args.limit,
        refresh_horses=args.refresh_horses, skip_horses=args.skip_horses)
    for y in converted:
        rebuild_csv(args.out, y)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jra_seiseki", description="JRA 成績表 PDF -> JSON/CSV")
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("update", help="download and convert new PDFs")
    up.add_argument("--year", type=int, nargs="*", help="years to process (default: this year)")
    up.add_argument("--out", type=Path, default=DEFAULT_OUT)
    up.add_argument("--force", action="store_true", help="re-convert files already in the manifest")
    up.add_argument("--limit", type=int, default=0, help="convert at most N files")
    up.add_argument("--wait", type=float, default=1.0, help="seconds between downloads")
    up.add_argument("--pdf-dir", help="keep downloaded PDFs here (reused on the next run)")
    up.set_defaults(func=cmd_update)

    pa = sub.add_parser("parse", help="convert local PDF(s) to JSON on stdout")
    pa.add_argument("pdf", nargs="+")
    pa.add_argument("--pretty", action="store_true")
    pa.set_defaults(func=cmd_parse)

    rb = sub.add_parser("rebuild-csv", help="regenerate yearly CSVs from the JSON files")
    rb.add_argument("--year", type=int, nargs="*")
    rb.add_argument("--out", type=Path, default=DEFAULT_OUT)
    rb.set_defaults(func=cmd_rebuild)

    pe = sub.add_parser("pedigree", help="fetch horse IDs and pedigrees from JRA 競走馬情報")
    pe.add_argument("--year", type=int, nargs="*", help="race years to cover (default: every converted year, newest first)")
    pe.add_argument("--out", type=Path, default=DEFAULT_OUT, help="seiseki data directory")
    pe.add_argument("--pedigree-out", type=Path, default=DEFAULT_PEDIGREE)
    pe.add_argument("--wait", type=float, default=0.5, help="minimum seconds between requests")
    pe.add_argument("--workers", type=int, default=2)
    pe.add_argument("--max-minutes", type=float, default=0, help="stop (saving progress) after N minutes")
    pe.add_argument("--limit", type=int, default=0, help="at most N races / N horses per run (testing)")
    pe.add_argument("--refresh-horses", action="store_true", help="re-fetch horses already stored")
    pe.add_argument("--skip-horses", action="store_true", help="only map races to horse IDs")
    pe.set_defaults(func=cmd_pedigree)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
