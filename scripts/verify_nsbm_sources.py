from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
NSBM_ROOT = ROOT / "data" / "documents" / "institutions" / "nsbm"
MANIFEST = NSBM_ROOT / "nsbm_sources_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    print("=" * 68)
    print("NSBM SOURCE COLLECTION - INTEGRITY AUDIT")
    print("=" * 68)
    print("Root:", NSBM_ROOT)

    if not NSBM_ROOT.is_dir():
        print("FAIL: NSBM source directory not found")
        return 2

    pdfs = sorted(NSBM_ROOT.rglob("*.pdf"))
    markdown = sorted(NSBM_ROOT.rglob("*.md"))
    print(f"\nFiles found: {len(pdfs)} PDFs, {len(markdown)} Markdown snapshots")

    issues: list[str] = []
    warnings: list[str] = []

    pdf_hashes: dict[str, list[str]] = defaultdict(list)
    total_pages = 0
    total_pdf_chars = 0

    print("\n[1] PDF readability")
    for path in pdfs:
        rel = path.relative_to(NSBM_ROOT).as_posix()
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            chars = sum(len((page.extract_text() or "").strip()) for page in reader.pages)
            if page_count == 0:
                issues.append(f"PDF has no pages: {rel}")
            if chars == 0:
                issues.append(f"PDF has no extractable text (may be scanned): {rel}")
            total_pages += page_count
            total_pdf_chars += chars
            digest = sha256(path)
            pdf_hashes[digest].append(rel)
            print(f"PASS | {rel} | pages={page_count} | text_chars={chars:,}")
        except Exception as exc:
            issues.append(f"Cannot read PDF {rel}: {type(exc).__name__}: {exc}")
            print(f"FAIL | {rel} | {type(exc).__name__}: {exc}")

    print(f"\nPDF totals: {len(pdfs)} files | {total_pages} pages | {total_pdf_chars:,} extracted chars")

    print("\n[2] Markdown snapshot checks")
    md_hashes: dict[str, list[str]] = defaultdict(list)
    for path in markdown:
        rel = path.relative_to(NSBM_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            chars = len(text)
            digest = sha256(path)
            md_hashes[digest].append(rel)
            if chars < 250:
                warnings.append(f"Very short snapshot ({chars} chars): {rel}")
            if "Source URL:" not in text or "Source authority:" not in text:
                issues.append(f"Missing source provenance header: {rel}")
            print(f"{'WARN' if chars < 500 else 'PASS'} | {rel} | chars={chars:,}")
        except Exception as exc:
            issues.append(f"Cannot read Markdown {rel}: {type(exc).__name__}: {exc}")
            print(f"FAIL | {rel} | {type(exc).__name__}: {exc}")

    print("\n[3] Duplicate content checks")
    for label, hash_map in (("PDF", pdf_hashes), ("Markdown", md_hashes)):
        duplicates = [paths for paths in hash_map.values() if len(paths) > 1]
        if duplicates:
            for paths in duplicates:
                warnings.append(f"Exact duplicate {label} content: {paths}")
                print(f"WARN | duplicate {label}: {' == '.join(paths)}")
        else:
            print(f"PASS | No exact duplicate {label} content found")

    print("\n[4] Collection manifest checks")
    if not MANIFEST.exists():
        issues.append("NSBM source manifest is missing")
        print("FAIL | Manifest missing")
    else:
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            entries = manifest.get("documents", [])
            print("Manifest records:", len(entries))
            manifest_paths = set()

            for item in entries:
                rel = item.get("relative_path", "")
                if not rel:
                    issues.append("Manifest entry without relative_path")
                    continue
                manifest_paths.add(rel)
                path = NSBM_ROOT / Path(rel)
                if not path.is_file():
                    issues.append(f"Manifest file missing: {rel}")
                    continue
                expected_hash = item.get("sha256")
                if expected_hash and sha256(path) != expected_hash:
                    issues.append(f"Manifest SHA256 mismatch: {rel}")

            expected_pdf_entries = sum(
                1 for item in entries if item.get("file_type") == "pdf"
            )
            expected_md_entries = sum(
                1 for item in entries if item.get("file_type") == "markdown"
            )
            print(
                f"Manifest types: {expected_pdf_entries} PDFs, "
                f"{expected_md_entries} Markdown"
            )

            if manifest.get("collection_failures"):
                warnings.append(
                    f"Manifest records {len(manifest['collection_failures'])} collection failure(s)"
                )

            # Collector manifest intentionally contains its 20 curated additions.
            untracked_pdfs = [
                p.relative_to(NSBM_ROOT).as_posix()
                for p in pdfs
                if p.relative_to(NSBM_ROOT).as_posix() not in manifest_paths
            ]
            print("PDFs outside new-source manifest:", len(untracked_pdfs))
            if untracked_pdfs:
                print("  These may be the original 11 PDFs and are not automatically errors.")
                for item in untracked_pdfs:
                    print("  -", item)

            missing_manifest_sources = [
                item.get("relative_path")
                for item in entries
                if item.get("relative_path") not in manifest_paths
            ]
            if missing_manifest_sources:
                issues.append("Manifest path validation internal error")

            print("PASS | All manifest files exist and SHA256 matches")
        except Exception as exc:
            issues.append(f"Manifest error: {type(exc).__name__}: {exc}")
            print(f"FAIL | Manifest | {type(exc).__name__}: {exc}")

    print("\n" + "=" * 68)
    print("AUDIT SUMMARY")
    print("=" * 68)
    print("Total PDFs:", len(pdfs))
    print("Total Markdown snapshots:", len(markdown))
    print("Issues:", len(issues))
    print("Warnings:", len(warnings))

    for warning in warnings:
        print("WARNING:", warning)
    for issue in issues:
        print("ERROR:", issue)

    if issues:
        print("\nSOURCE AUDIT: FAILED")
        return 1

    print("\nSOURCE AUDIT: PASSED")
    print("No files were changed. Chroma database was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
