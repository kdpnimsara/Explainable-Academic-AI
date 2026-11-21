from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NSBM_ROOT = ROOT / "data" / "documents" / "institutions" / "nsbm"
MANIFEST = NSBM_ROOT / "nsbm_sources_manifest.json"
COLLECTOR = ROOT / "scripts" / "collect_nsbm_sources.py"
AUDITOR = ROOT / "scripts" / "verify_nsbm_sources.py"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    print("=" * 68)
    print("NSBM MANIFEST HASH REPAIR")
    print("=" * 68)

    if not MANIFEST.is_file():
        print("ERROR: Manifest not found:", MANIFEST)
        return 2

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = data.get("documents", [])
    repaired = 0
    unchanged = 0
    missing = []

    for item in entries:
        relative = item.get("relative_path")
        if not relative:
            missing.append("<entry without relative_path>")
            continue

        target = NSBM_ROOT / Path(relative)
        if not target.is_file():
            missing.append(relative)
            continue

        actual = file_hash(target)
        if item.get("sha256") != actual:
            item["sha256"] = actual
            repaired += 1
        else:
            unchanged += 1

        item["file_size_bytes"] = target.stat().st_size

    if missing:
        print("ERROR: Manifest files missing:")
        for name in missing:
            print(" -", name)
        return 2

    # The collector previously hashed the in-memory Markdown string.
    # Windows translates LF to CRLF when Path.write_text() writes the file.
    # Hash the exact bytes on disk instead.
    if not COLLECTOR.is_file():
        print("ERROR: Collector script not found:", COLLECTOR)
        return 2

    collector_text = COLLECTOR.read_text(encoding="utf-8")
    old_line = (
        'digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()'
    )
    new_line = 'digest = hashlib.sha256(target.read_bytes()).hexdigest()'

    if old_line in collector_text:
        collector_text = collector_text.replace(old_line, new_line)
        COLLECTOR.write_text(collector_text, encoding="utf-8", newline="\n")
        print("Collector fix: applied")
    elif new_line in collector_text:
        print("Collector fix: already applied")
    else:
        print("ERROR: Could not safely locate collector hash expression.")
        print("No further changes made.")
        return 2

    MANIFEST.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("Manifest records:", len(entries))
    print("Hashes repaired:", repaired)
    print("Hashes already correct:", unchanged)
    print("Missing files:", len(missing))

    # Report short extracted page bodies for manual review before ingestion.
    short_pages = [
        item for item in entries
        if item.get("file_type") == "markdown"
        and int(item.get("character_count", 0) or 0) < 500
    ]
    print("\nShort website snapshots to review before indexing:")
    if short_pages:
        for item in short_pages:
            print(
                f" - {item.get('relative_path')} "
                f"({item.get('character_count')} source chars)"
            )
    else:
        print("None")

    print("\nRunning source integrity audit...")
    if not AUDITOR.is_file():
        print("ERROR: Audit script not found:", AUDITOR)
        return 2

    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(COLLECTOR), str(AUDITOR)],
        cwd=ROOT,
    )
    if compile_result.returncode != 0:
        print("ERROR: Python syntax check failed.")
        return compile_result.returncode

    audit_result = subprocess.run(
        [sys.executable, str(AUDITOR)],
        cwd=ROOT,
    )
    if audit_result.returncode != 0:
        print("\nSOURCE AUDIT: STILL NEEDS ATTENTION")
        return audit_result.returncode

    print("\nMANIFEST REPAIR: PASSED")
    print("No source document content was changed.")
    print("Chroma database was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
