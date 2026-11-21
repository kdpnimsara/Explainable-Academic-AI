#!/usr/bin/env python3
"""Incrementally collect public SLIIT knowledge-base sources.

- Scans official SLIIT sitemaps and selected public SLIIT pages.
- Downloads public PDFs and snapshots useful public HTML pages as Markdown.
- Adds only new sources; skips URLs already in the manifest when their local
  files still exist.
- Preserves all existing SLIIT files and the other university/vector data.
- Respects robots.txt and does not log in or bypass access restrictions.

This is a best-effort collector for publicly accessible sources, not a copy of
private CourseWeb, student portal, or library account content.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.robotparser
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIIT_DIR = ROOT / "data" / "documents" / "institutions" / "sliit"
MANIFEST = SLIIT_DIR / "sliit_sources_manifest.json"
USER_AGENT = "AcademicKnowledgeBaseCollector/1.3 (public educational sources; incremental; respectful crawl)"
MAX_BYTES = 90 * 1024 * 1024
REQUEST_DELAY = 0.30
MAX_NEW_PAGES = 280
MAX_NEW_PDFS = 180

FOLDERS = (
    "academic_calendars", "academic_regulations", "examinations",
    "fees_and_scholarships", "official_notices", "programmes_and_faculties",
    "quality_assurance", "research_and_projects", "student_services",
)

# Useful public entry points. The collector discovers further links and sitemap
# entries from these. Explicit categories avoid ambiguous generic page titles.
SEED_PAGES: list[tuple[str, str | None]] = [
    ("https://www.sliit.lk/", "official_notices"),
    ("https://www.sliit.lk/international-students/downloads", "student_services"),
    ("https://www.sliit.lk/academic/academic-quality-and-regulations", "quality_assurance"),
    ("https://www.sliit.lk/study/fees-and-scholarships/fees-and-costs", "fees_and_scholarships"),
    ("https://www.sliit.lk/study/fees-and-scholarships/scholarships", "fees_and_scholarships"),
    ("https://www.sliit.lk/study/fees-and-scholarships", "fees_and_scholarships"),
    ("https://www.sliit.lk/computing/programmes", "programmes_and_faculties"),
    ("https://www.sliit.lk/engineering/programmes/", "programmes_and_faculties"),
    ("https://www.sliit.lk/business/programmes/", "programmes_and_faculties"),
    ("https://www.sliit.lk/humanities-sciences/", "programmes_and_faculties"),
    ("https://www.sliit.lk/architecture/", "programmes_and_faculties"),
    ("https://www.sliit.lk/graduate-studies-research/", "programmes_and_faculties"),
    ("https://www.sliit.lk/about/governance/university-structure", "academic_regulations"),
    ("https://www.sliit.lk/life-at-sliit/support-services", "student_services"),
    ("https://www.sliit.lk/research", "research_and_projects"),
    ("https://support.sliit.lk/kb/articles/semester-registration", "academic_regulations"),
    ("https://support.sliit.lk/kb/articles/general-timelines-for-semester-module-registration-and-examination", "academic_calendars"),
    ("https://support.sliit.lk/kb/articles/rules-and-regulations-for-students", "academic_regulations"),
    ("https://support.sliit.lk/kb/articles/useful-online-resources-for-sliit-students", "student_services"),
    # Public CourseWeb catalogue pages only. Individual course materials may
    # require student sign-in and are not accessed by this collector.
    ("https://courseweb.sliit.lk/course/index.php?categoryid=2", "examinations"),
    ("https://courseweb.sliit.lk/course/index.php?categoryid=2117", "academic_calendars"),
]

# Current official PDFs. Existing URLs are skipped automatically. Older local
# copies (for example, the 2025 guide) are left untouched.
SEED_PDFS: list[tuple[str, str, str | None]] = [
    ("https://static.sliit.lk/wp-content/uploads/downloads/guides-booklet/sliit-student-guide-2026-v1.pdf", "SLIIT Student Guide 2026", "student_services"),
    ("https://static.sliit.lk/wp-content/uploads/downloads/guides-booklet/Computing-Faculty-Booklet-2026-V1.pdf", "SLIIT Faculty of Computing Student Guide 2026", "programmes_and_faculties"),
    ("https://static.sliit.lk/wp-content/uploads/downloads/Fee-Structure-2026-V2.pdf", "SLIIT Fee Structure 2026", "fees_and_scholarships"),
]

SITEMAP_CANDIDATES = [
    "https://www.sliit.lk/robots.txt",
    "https://www.sliit.lk/sitemap.xml",
    "https://www.sliit.lk/images-sitemap.xml",
    "https://sliit.lk/sitemap.xml",
]

PAGE_KEYWORDS = (
    "academic", "calendar", "semester", "exam", "examination", "assessment",
    "fee", "scholarship", "financial-assistance", "programme", "program",
    "faculty", "department", "degree", "student", "library", "research",
    "thesis", "quality", "regulation", "policy", "notice", "career",
    "support", "admission", "handbook", "guide", "international", "school",
    "governance", "campus", "service", "welfare", "registration", "rules",
    "downloads", "brochure", "prospectus", "faq", "integrity", "accreditation",
)
PDF_KEYWORDS = PAGE_KEYWORDS + ("pdf", "booklet", "manual", "guideline", "prospectus")
BLOCKED_PATH_PARTS = (
    "/login", "/logout", "/my/", "/dashboard", "/student-portal",
    "/wp-admin", "/mod/", "/user/", "/account", "/reset-password",
    "/private/", "/profile/", "/my-courses", "/course/view.php?id=",
)
ROBOTS: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    """Normalize fragments/trailing slash and tracking query parameters."""
    p = urllib.parse.urlsplit(urllib.parse.urldefrag(url)[0])
    host = (p.hostname or "").lower()
    netloc = host
    if p.port and p.port not in (80, 443):
        netloc = f"{host}:{p.port}"
    path = p.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
    query_items = [(k, v) for k, v in query_items if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    query = urllib.parse.urlencode(sorted(query_items))
    return urllib.parse.urlunsplit((p.scheme.lower(), netloc, path, query, ""))


def allowed_url(url: str) -> bool:
    try:
        p = urllib.parse.urlsplit(url)
        host = (p.hostname or "").lower()
        if p.scheme not in ("http", "https"):
            return False
        if not (host == "sliit.lk" or host.endswith(".sliit.lk")):
            return False
        if host.startswith("uattest."):
            return False
        pathq = ((p.path or "/") + ("?" + p.query if p.query else "")).lower()
        if any(part in pathq for part in BLOCKED_PATH_PARTS):
            return False
        return True
    except Exception:
        return False


def robots_allowed(url: str) -> bool:
    p = urllib.parse.urlsplit(url)
    origin = f"{p.scheme}://{p.netloc}"
    if origin not in ROBOTS:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(origin + "/robots.txt")
        try:
            rp.read()
            ROBOTS[origin] = rp
        except Exception:
            ROBOTS[origin] = None
    rp = ROBOTS[origin]
    return True if rp is None else rp.can_fetch(USER_AGENT, url)


def category_for(url: str, title: str = "", forced: str | None = None) -> str:
    if forced in FOLDERS:
        return forced
    s = (url + " " + title).lower().replace("_", "-")
    if any(x in s for x in ("calendar", "semester-dates", "academic-year", "general-timelines", "registration-dates")):
        return "academic_calendars"
    if any(x in s for x in ("exam", "examination", "assessment", "result", "resit", "invigil", "timetable")):
        return "examinations"
    if any(x in s for x in ("fee", "scholarship", "financial-aid", "financial-assistance", "payment", "bursar", "loan")):
        return "fees_and_scholarships"
    if any(x in s for x in ("quality", "accreditation", "iqau", "audit", "quality-assurance", "ranking")):
        return "quality_assurance"
    if any(x in s for x in ("research", "thesis", "publication", "innovation", "ethics", "research-award", "research-grant", "journal", "conference")):
        return "research_and_projects"
    if any(x in s for x in ("regulation", "by-law", "bylaw", "policy", "rules", "disciplinary", "governance", "integrity", "semester-registration")):
        return "academic_regulations"
    if any(x in s for x in ("student", "library", "support", "career", "welfare", "service", "elibrary", "wellbeing", "clubs", "societies", "health", "faq", "international-students")):
        return "student_services"
    if any(x in s for x in ("programme", "program", "faculty", "degree", "department", "school", "guide", "brochure", "prospectus", "admission", "computing", "engineering", "business", "humanities", "science", "architecture", "nursing", "law", "course")):
        return "programmes_and_faculties"
    return "official_notices"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.skip_depth = 0
        self.skip_stack: list[str] = []
        self.anchor_href: str | None = None
        self.anchor_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag in ("script", "style", "noscript", "svg", "form"):
            self.skip_stack.append(tag)
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        if tag == "a" and attrs.get("href"):
            self.anchor_href = attrs["href"]
            self.anchor_parts = []
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if self.skip_stack and tag == self.skip_stack[-1]:
                self.skip_stack.pop()
                self.skip_depth -= 1
            return
        if tag == "title":
            self.in_title = False
        if tag == "a" and self.anchor_href is not None:
            label = " ".join(" ".join(self.anchor_parts).split())
            self.links.append((self.anchor_href, label))
            self.anchor_href = None
            self.anchor_parts = []
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"):
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        self.text_parts.append(clean + " ")
        if self.in_title:
            self.title_parts.append(clean)
        if self.anchor_href is not None:
            self.anchor_parts.append(clean)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def text(self) -> str:
        lines = [re.sub(r"[ \t]+", " ", x).strip() for x in "".join(self.text_parts).splitlines()]
        lines = [x for x in lines if x]
        out: list[str] = []
        for line in lines:
            if not out or out[-1] != line:
                out.append(line)
        return "\n".join(out)


def fetch(url: str) -> tuple[bytes, str, str]:
    if not robots_allowed(url):
        raise PermissionError("blocked by robots.txt")
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/pdf,application/xml,text/plain,*/*",
    })
    retryable_http = {408, 425, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=40) as response:
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError(f"file exceeds {MAX_BYTES // (1024 * 1024)} MB safety limit")
                return data, content_type, final_url
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in retryable_http or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt == 3:
                raise
        pause = attempt * 2
        print(f"[RETRY {attempt}/2] {url}: {last_error}; retrying in {pause}s")
        time.sleep(pause)
    assert last_error is not None
    raise last_error


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sitemap_locations(data: bytes) -> list[str]:
    try:
        root = ET.fromstring(data)
        return [x.text.strip() for x in root.findall(".//{*}loc") if x.text and x.text.strip()]
    except ET.ParseError:
        return []


def discover_sitemap_urls() -> list[str]:
    pages: list[str] = []
    visited: set[str] = set()
    queue = list(SITEMAP_CANDIDATES)
    while queue and len(visited) < 30:
        url = queue.pop(0)
        key = normalize_url(url)
        if key in visited or not allowed_url(url):
            continue
        visited.add(key)
        try:
            data, _, final_url = fetch(url)
            if not allowed_url(final_url):
                continue
            if "robots.txt" in url.lower():
                for line in data.decode("utf-8", errors="replace").splitlines():
                    if line.lower().startswith("sitemap:"):
                        candidate = line.split(":", 1)[1].strip()
                        if allowed_url(candidate):
                            queue.append(candidate)
            locs = sitemap_locations(data)
            for loc in locs:
                if not allowed_url(loc):
                    continue
                low = loc.lower()
                if low.endswith((".xml", ".xml.gz")) and ("sitemap" in low or "site-map" in low):
                    queue.append(loc)
                elif is_relevant_page(loc) or is_pdf_candidate(loc):
                    pages.append(loc)
            if locs:
                print(f"[SITEMAP] {url} -> {len(locs)} entries")
        except Exception as exc:
            print(f"[SITEMAP SKIP] {url}: {exc}")
        time.sleep(REQUEST_DELAY)
    return list(dict.fromkeys(pages))


def is_relevant_page(url: str, label: str = "") -> bool:
    low = (url + " " + label).lower()
    if any(x in low for x in ("/tag/", "/author/", "/category/", "wp-json", "feed=")):
        return False
    return any(k in low for k in PAGE_KEYWORDS)


def is_pdf_candidate(url: str, label: str = "") -> bool:
    low = (url + " " + label).lower()
    path = urllib.parse.urlsplit(url).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path or ("pdf" in label.lower() and any(k in low for k in PDF_KEYWORDS))


def safe_filename(url: str, suffix: str) -> str:
    path_part = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1])
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", path_part or "sliit_source").strip("._-")
    for ext in (".pdf", ".html", ".htm"):
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    if not base:
        base = "sliit_source"
    digest = hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:10]
    return f"{base[:125]}_{digest}{suffix}"


def existing_file(record: dict) -> bool:
    rel = record.get("local_path")
    if not rel:
        return False
    try:
        path = (ROOT / Path(rel)).resolve()
        return path.is_relative_to(ROOT.resolve()) and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def load_manifest() -> tuple[dict, dict[str, dict]]:
    if not MANIFEST.exists():
        base = {
            "institution_id": "SLIIT",
            "institution_name": "Sri Lanka Institute of Information Technology",
            "manifest_version": "1.2",
            "sources": [],
        }
        return base, {}
    try:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("sources", []), list):
            raise ValueError("unexpected manifest structure")
        records = {}
        for item in payload.get("sources", []):
            if isinstance(item, dict) and item.get("source_url"):
                records[normalize_url(item["source_url"])] = item
        return payload, records
    except Exception as exc:
        raise RuntimeError(f"Cannot safely read existing manifest; stopping without overwriting it: {exc}")


def write_new_file(category: str, filename: str, data: bytes) -> Path:
    folder = SLIIT_DIR / category
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    if target.exists():
        # Do not overwrite an existing file. Use an additional digest suffix.
        stem, suffix = target.stem, target.suffix
        target = folder / f"{stem}_new_{sha256(data)[:8]}{suffix}"
    target.write_bytes(data)
    return target


def add_record(records: dict[str, dict], record: dict) -> None:
    records[normalize_url(record["source_url"])] = record


def main() -> int:
    SLIIT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in FOLDERS:
        (SLIIT_DIR / folder).mkdir(parents=True, exist_ok=True)

    manifest, records = load_manifest()
    original_count = len(records)
    existing_count = sum(1 for x in records.values() if existing_file(x))
    new_pages = new_pdfs = skipped_existing = page_attempts = 0
    errors: list[dict] = []

    print("=" * 72)
    print("SLIIT PUBLIC KNOWLEDGE-BASE COLLECTOR V1.3 — INCREMENTAL")
    print("=" * 72)
    print(f"Project folder       : {SLIIT_DIR}")
    print(f"Manifest sources     : {original_count}")
    print(f"Existing local files : {existing_count}")
    print("Mode                 : discover + add-new-only + categorize + manifest")
    print("Access               : public official SLIIT sources; robots.txt respected")
    print("Existing files       : not overwritten or deleted")
    print("Private/login data   : not accessed\n")

    sitemap_urls = discover_sitemap_urls()
    page_queue: list[tuple[str, str | None]] = list(SEED_PAGES)
    pdf_queue: list[tuple[str, str, str | None]] = list(SEED_PDFS)
    for url in sitemap_urls:
        if is_pdf_candidate(url):
            pdf_queue.append((url, "PDF discovered in SLIIT public sitemap", None))
        else:
            page_queue.append((url, None))

    # Process only new relevant pages. Existing manifest URLs with local files
    # are skipped before making a network request.
    seen_pages: set[str] = set()
    discovered_pdf_keys: set[str] = set()
    while page_queue and page_attempts < MAX_NEW_PAGES:
        url, forced_category = page_queue.pop(0)
        url = urllib.parse.urldefrag(url)[0]
        key = normalize_url(url)
        if key in seen_pages or not allowed_url(url):
            continue
        seen_pages.add(key)
        old = records.get(key)
        if old and existing_file(old):
            skipped_existing += 1
            continue
        if not forced_category and not is_relevant_page(url):
            continue
        page_attempts += 1
        try:
            data, content_type, final_url = fetch(url)
            if not allowed_url(final_url):
                print(f"[REDIRECT BLOCKED] {url} -> {final_url}")
                continue
            if data.startswith(b"%PDF-") or "pdf" in content_type:
                pdf_queue.append((final_url, "PDF returned from a public SLIIT page", forced_category))
                continue
            if "html" not in content_type and not data.lstrip().startswith((b"<", b"<!DOCTYPE")):
                continue
            parser = PageParser()
            parser.feed(data.decode("utf-8", errors="replace"))
            title = parser.title or urllib.parse.urlsplit(final_url).path.strip("/") or "SLIIT Official Page"
            body = parser.text
            if len(body) >= 160:
                final_key = normalize_url(final_url)
                old_final = records.get(final_key)
                if old_final and existing_file(old_final):
                    skipped_existing += 1
                else:
                    category = category_for(final_url, title, forced_category)
                    filename = safe_filename(final_url, ".md")
                    payload = (
                        f"# {title}\n\nInstitution: Sri Lanka Institute of Information Technology (SLIIT)\n\n"
                        f"Source URL: {final_url}\n\nCollected (UTC): {utc_now()}\n\n"
                        "Source type: Public SLIIT website snapshot\n\n---\n\n"
                        f"{body}\n"
                    ).encode("utf-8")
                    output = write_new_file(category, filename, payload)
                    add_record(records, {
                        "title": title,
                        "category": category,
                        "format": "markdown_snapshot",
                        "source_url": final_url,
                        "local_path": output.relative_to(ROOT).as_posix(),
                        "sha256": sha256(payload),
                        "bytes": len(payload),
                        "status": "downloaded",
                        "collected_at_utc": utc_now(),
                    })
                    new_pages += 1
                    print(f"[NEW PAGE {new_pages}] {category}: {title[:85]}")

            for href, label in parser.links:
                absolute = urllib.parse.urldefrag(urllib.parse.urljoin(final_url, href))[0]
                if not allowed_url(absolute):
                    continue
                if is_pdf_candidate(absolute, label):
                    pkey = normalize_url(absolute)
                    if pkey not in discovered_pdf_keys:
                        pdf_queue.append((absolute, label or "PDF linked from SLIIT page", None))
                        discovered_pdf_keys.add(pkey)
                elif is_relevant_page(absolute, label):
                    if normalize_url(absolute) not in seen_pages:
                        page_queue.append((absolute, None))
        except Exception as exc:
            errors.append({"url": url, "stage": "page", "error": str(exc)})
            print(f"[PAGE SKIP] {url}: {exc}")
        time.sleep(REQUEST_DELAY)

    # Download newly discovered PDFs; never rewrite a PDF whose source URL is
    # already represented by an existing local file.
    seen_pdfs: set[str] = set()
    for url, label, forced_category in pdf_queue:
        if new_pdfs >= MAX_NEW_PDFS:
            break
        url = urllib.parse.urldefrag(url)[0]
        key = normalize_url(url)
        if key in seen_pdfs or not allowed_url(url):
            continue
        seen_pdfs.add(key)
        old = records.get(key)
        if old and existing_file(old):
            skipped_existing += 1
            continue
        try:
            data, content_type, final_url = fetch(url)
            if not allowed_url(final_url):
                print(f"[PDF REDIRECT BLOCKED] {url} -> {final_url}")
                continue
            if not data.startswith(b"%PDF-"):
                if "html" in content_type or data.lstrip().startswith(b"<"):
                    # HTML article; it may be discovered in the page pass only
                    print(f"[PDF LINK IS HTML] {url}")
                    continue
                raise ValueError(f"response is not a PDF (content-type: {content_type or 'unknown'})")
            final_key = normalize_url(final_url)
            old_final = records.get(final_key)
            if old_final and existing_file(old_final):
                skipped_existing += 1
                continue
            category = category_for(final_url, label, forced_category)
            filename = safe_filename(final_url, ".pdf")
            output = write_new_file(category, filename, data)
            title = label or urllib.parse.unquote(urllib.parse.urlsplit(final_url).path.rsplit("/", 1)[-1])
            add_record(records, {
                "title": title,
                "category": category,
                "format": "pdf",
                "source_url": final_url,
                "local_path": output.relative_to(ROOT).as_posix(),
                "sha256": sha256(data),
                "bytes": len(data),
                "content_type": content_type,
                "status": "downloaded",
                "collected_at_utc": utc_now(),
            })
            new_pdfs += 1
            print(f"[NEW PDF {new_pdfs}] {category}: {filename} ({len(data):,} bytes)")
        except Exception as exc:
            errors.append({"url": url, "stage": "pdf", "error": str(exc)})
            print(f"[PDF SKIP] {url}: {exc}")
        time.sleep(REQUEST_DELAY)

    final_records = sorted(records.values(), key=lambda x: (x.get("category", ""), x.get("title", "").lower()))
    manifest.update({
        "institution_id": "SLIIT",
        "institution_name": "Sri Lanka Institute of Information Technology",
        "manifest_version": "1.3",
        "generated_at_utc": utc_now(),
        "collection_scope": "Publicly accessible SLIIT-owned webpages and files only",
        "collection_method": "Incremental collection using official seed pages, public sitemap discovery, and relevant public page links",
        "note": "Best-effort public-source collection; not a complete copy of login-only or private institutional systems. Existing files are skipped when already present locally.",
        "source_count": len(final_records),
        "pdf_count": sum(1 for x in final_records if x.get("format") == "pdf"),
        "page_snapshot_count": sum(1 for x in final_records if x.get("format") == "markdown_snapshot"),
        "new_sources_this_run": new_pages + new_pdfs,
        "existing_sources_skipped_this_run": skipped_existing,
        "sources": final_records,
        "collection_errors": errors,
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SLIIT INCREMENTAL COLLECTION SUMMARY")
    print("=" * 72)
    print(f"New HTML pages       : {new_pages}")
    print(f"New PDFs             : {new_pdfs}")
    print(f"New sources total    : {new_pages + new_pdfs}")
    print(f"Existing source skips: {skipped_existing}")
    print(f"Manifest total       : {len(final_records)}")
    print(f"Fetch warnings       : {len(errors)}")
    print(f"Manifest             : {MANIFEST.relative_to(ROOT)}")
    print("\nAll sources by category:")
    for category in FOLDERS:
        count = sum(1 for item in final_records if item.get("category") == category)
        print(f"  {category:26s}: {count}")
    if errors:
        print("\nSome public sources could not be fetched. See collection_errors in the manifest.")
    print("No pre-existing files were overwritten or deleted.")
    print("Collection finished. Review dates and source quality before RAG ingestion.")
    return 0 if final_records else 1


if __name__ == "__main__":
    sys.exit(main())
