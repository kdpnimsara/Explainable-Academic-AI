from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
INSTITUTION_DIR = ROOT / "data" / "documents" / "institutions" / "sliit"
MANIFEST_PATH = INSTITUTION_DIR / "sliit_sources_manifest.json"
DOWNLOADS_PAGE = "https://www.sliit.lk/study/downloads"
ALLOWED_HOST = "webasset.sliit.lk"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
MAX_BYTES = 100 * 1024 * 1024


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(html.unescape(value.strip()))
                break


def category_for(filename: str) -> str:
    name = filename.lower()
    if "fee-structure" in name or "fee" in name:
        return "fees_and_scholarships"
    if "application-mphil" in name or "research" in name or "phd" in name:
        return "research_and_projects"
    if any(term in name for term in ("student-guide", "student_guide", "international-student", "application_", "application-form")):
        return "student_services"
    return "programmes_and_faculties"


def title_for(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_\d{9,}$", "", stem)
    stem = re.sub(r"[-_]+", " ", stem).strip()
    if not stem.lower().startswith("sliit"):
        stem = "SLIIT " + stem
    return stem


def fetch(url: str, timeout: int = 45) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"})
    with urlopen(req, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"HTTP status {status}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                raise RuntimeError(f"File exceeds {MAX_BYTES // (1024 * 1024)} MB limit")
            chunks.append(chunk)
        return b"".join(chunks)


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: SLIIT manifest not found: {MANIFEST_PATH}")
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
        print("ERROR: Unexpected SLIIT manifest structure; no changes made.")
        return 2

    print("=" * 72)
    print("SLIIT PUBLIC DOWNLOADS-PAGE PDF COLLECTOR")
    print("=" * 72)
    print(f"Project: {INSTITUTION_DIR}")
    print("Source: the publicly linked SLIIT Downloads page")
    print("Scope: exact PDF links published on that page; no login-only records")
    print("Existing files: preserved; existing source URLs are skipped")

    try:
        page_bytes = fetch(DOWNLOADS_PAGE)
        page_html = page_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"ERROR: Could not fetch SLIIT downloads page: {exc}")
        return 1

    parser = AnchorParser()
    parser.feed(page_html)
    urls: list[str] = []
    for href in parser.hrefs:
        absolute = urljoin(DOWNLOADS_PAGE, href)
        parsed = urlparse(absolute)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            continue
        if not parsed.path.lower().endswith(".pdf"):
            continue
        if absolute not in urls:
            urls.append(absolute)

    if not urls:
        print("ERROR: No SLIIT-hosted PDF links were found on the official downloads page.")
        return 1

    print(f"Official PDF links discovered: {len(urls)}")

    source_urls = {str(item.get("source_url", "")) for item in manifest["sources"]}
    existing_paths = {str(item.get("local_path", "")).replace("\\", "/") for item in manifest["sources"]}
    backup_dir = ROOT / "data" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"sliit_manifest_before_public_pdf_download_{stamp}.json"
    backup_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    added: list[dict] = []
    errors: list[dict] = []
    skipped = 0

    for index, url in enumerate(urls, start=1):
        if url in source_urls:
            skipped += 1
            print(f"[SKIP {index}/{len(urls)}] Already in manifest: {Path(urlparse(url).path).name}")
            continue

        filename = unquote(Path(urlparse(url).path).name)
        category = category_for(filename)
        relative = f"data/documents/institutions/sliit/{category}/{filename}"
        normalized_relative = relative.replace("\\", "/")
        destination = ROOT / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if normalized_relative in existing_paths:
            skipped += 1
            print(f"[SKIP {index}/{len(urls)}] Local path already registered: {filename}")
            continue

        if destination.exists():
            # Never overwrite an existing file whose provenance is not this URL.
            suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
            destination = destination.with_name(f"{destination.stem}_{suffix}{destination.suffix}")
            relative = destination.relative_to(ROOT).as_posix()

        data = None
        last_error = None
        for attempt in range(1, 4):
            try:
                data = fetch(url)
                break
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(attempt * 2)

        if data is None:
            message = str(last_error or "unknown download error")
            errors.append({"url": url, "stage": "pdf_download", "error": message})
            print(f"[FAILED {index}/{len(urls)}] {filename}: {message}")
            continue

        if len(data) < 5 or not data.startswith(b"%PDF-"):
            errors.append({"url": url, "stage": "pdf_validation", "error": "Response was not a valid PDF signature"})
            print(f"[INVALID {index}/{len(urls)}] {filename}: response is not a PDF")
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
        record = {
            "title": title_for(filename),
            "category": category,
            "format": "pdf",
            "source_url": url,
            "local_path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "status": "downloaded",
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        added.append(record)
        source_urls.add(url)
        existing_paths.add(relative.replace("\\", "/"))
        print(f"[DOWNLOADED {index}/{len(urls)}] {filename} ({len(data):,} bytes) -> {category}")

    manifest["sources"] = list(manifest["sources"]) + added
    manifest["collection_errors"] = list(manifest.get("collection_errors", [])) + errors
    manifest["source_count"] = len(manifest["sources"])
    manifest["pdf_count"] = sum(1 for item in manifest["sources"] if str(item.get("local_path", "")).lower().endswith(".pdf"))
    manifest["page_snapshot_count"] = manifest["source_count"] - manifest["pdf_count"]
    manifest["new_sources_this_run"] = len(added)
    manifest["existing_sources_skipped_this_run"] = skipped
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    temporary_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(MANIFEST_PATH)

    print("\n" + "=" * 72)
    print("SLIIT PUBLIC PDF DOWNLOAD SUMMARY")
    print("=" * 72)
    print(f"PDF links discovered : {len(urls)}")
    print(f"New PDFs downloaded  : {len(added)}")
    print(f"Existing sources skip: {skipped}")
    print(f"Download failures    : {len(errors)}")
    print(f"Manifest total        : {manifest['source_count']}")
    print(f"Manifest PDF count    : {manifest['pdf_count']}")
    print(f"Manifest snapshots    : {manifest['page_snapshot_count']}")
    print(f"Manifest              : {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Manifest backup       : {backup_path.relative_to(ROOT)}")
    if errors:
        print("Some links could not be fetched; details were appended to collection_errors.")
    print("No existing project documents were overwritten or deleted.")
    return 0 if added or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
