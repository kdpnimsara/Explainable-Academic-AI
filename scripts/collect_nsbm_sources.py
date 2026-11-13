from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from collections import Counter
import hashlib
import json
import re
import sys

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
NSBM_ROOT = ROOT / "data" / "documents" / "institutions" / "nsbm"
MANIFEST_PATH = NSBM_ROOT / "nsbm_sources_manifest.json"
TODAY = datetime.now().astimezone().date().isoformat()
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/120 Safari/537.36"
)


# Official NSBM PDFs. The thesis guideline is already downloaded locally;
# the script validates it and downloads it again only if it is invalid.
PDF_SOURCES = [
    {
        "filename": "NSBM_Thesis_Preparation_and_Formatting_Guidelines.pdf",
        "relative_path": "research_and_projects/NSBM_Thesis_Preparation_and_Formatting_Guidelines.pdf",
        "url": "https://library.nsbm.ac.lk/Uploads/systemuploads/Thesis%20Guideline%20-%20Senate%20Approved.pdf",
        "title": "NSBM Thesis Preparation and Formatting Guidelines",
        "category": "thesis_guidelines",
        "version": "Senate-approved guideline; website snapshot 2026",
        "authority": "NSBM Green University Library (official research support)",
        "note": "Undergraduate and postgraduate thesis preparation and formatting guidance.",
    },
    {
        "filename": "NSBM_Oracle_2026_May.pdf",
        "relative_path": "programmes_and_faculties/faculty_of_computing_bulletins/NSBM_Oracle_2026_May.pdf",
        "url": "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2026/06/Oracle-May-2.pdf",
        "title": "The Oracle – Faculty of Computing Monthly Bulletin, May 2026",
        "category": "faculty_bulletin",
        "version": "May 2026",
        "authority": "NSBM Faculty of Computing official monthly bulletin; informational, not a regulation",
        "note": "Supplementary faculty news and industry context; not an academic policy source.",
    },
    {
        "filename": "NSBM_Oracle_2026_March.pdf",
        "relative_path": "programmes_and_faculties/faculty_of_computing_bulletins/NSBM_Oracle_2026_March.pdf",
        "url": "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2026/03/Oracle-March-Issue-2026.pdf",
        "title": "The Oracle – Faculty of Computing Monthly Bulletin, March 2026",
        "category": "faculty_bulletin",
        "version": "March 2026",
        "authority": "NSBM Faculty of Computing official monthly bulletin; informational, not a regulation",
        "note": "Supplementary faculty news and industry context; not an academic policy source.",
    },
    {
        "filename": "NSBM_Oracle_2026_February.pdf",
        "relative_path": "programmes_and_faculties/faculty_of_computing_bulletins/NSBM_Oracle_2026_February.pdf",
        "url": "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2026/02/The-Oracle_Feb2026Issue-2.pdf",
        "title": "The Oracle – Faculty of Computing Monthly Bulletin, February 2026",
        "category": "faculty_bulletin",
        "version": "February 2026",
        "authority": "NSBM Faculty of Computing official monthly bulletin; informational, not a regulation",
        "note": "Supplementary faculty news and industry context; not an academic policy source.",
    },
    {
        "filename": "NSBM_Oracle_2026_January.pdf",
        "relative_path": "programmes_and_faculties/faculty_of_computing_bulletins/NSBM_Oracle_2026_January.pdf",
        "url": "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2026/01/Oracle_January2026.pdf",
        "title": "The Oracle – Faculty of Computing Monthly Bulletin, January 2026",
        "category": "faculty_bulletin",
        "version": "January 2026",
        "authority": "NSBM Faculty of Computing official monthly bulletin; informational, not a regulation",
        "note": "Supplementary faculty news and industry context; not an academic policy source.",
    },
]


# Pages are saved as dated Markdown snapshots with the original official URL.
# Dynamic pages are snapshots only; they are not treated as live services.
PAGE_SOURCES = [
    {
        "path": "programmes_and_faculties/NSBM_BSc_Hons_Data_Science_Programme.md",
        "url": "https://www.nsbm.ac.lk/course/bsc-honours-in-data-science/",
        "title": "NSBM BSc (Honours) in Data Science – Official Programme Page",
        "category": "programme_profile",
        "authority": "NSBM Green University official programme page; programme information, not a substitute for the approved curriculum",
        "version": "Website snapshot " + TODAY,
        "note": (
            "The public page describes the four-year programme, year-wise modules, "
            "mandatory third-year industry placement and fourth-year research project. "
            "Course-fee lines are intentionally omitted from this snapshot because the "
            "same page displays inconsistent fee formats; confirm all current fees with Admissions."
        ),
        "remove_programme_fee_section": True,
    },
    {
        "path": "programmes_and_faculties/NSBM_Department_of_Computer_and_Data_Science.md",
        "url": "https://www.nsbm.ac.lk/department-of-data-science/",
        "title": "NSBM Department of Computer and Data Science – Official Page",
        "category": "faculty_information",
        "authority": "NSBM Faculty of Computing official department page; general information",
        "version": "Website snapshot " + TODAY,
        "note": "Department profile and public programme information. Staff and contact details can change.",
    },
    {
        "path": "programmes_and_faculties/NSBM_Faculty_of_Computing.md",
        "url": "https://www.nsbm.ac.lk/faculty-of-computing/",
        "title": "NSBM Faculty of Computing – Official Page",
        "category": "faculty_information",
        "authority": "NSBM Faculty of Computing official page; general information",
        "version": "Website snapshot " + TODAY,
        "note": "Includes public faculty information and links to The Oracle bulletins. Not an academic regulation.",
    },
    {
        "path": "programmes_and_faculties/NSBM_Faculty_of_Computing_Curriculum_Revision_2025_Notice.md",
        "url": "https://www.nsbm.ac.lk/faculty-of-computing-successfully-concludes-the-curriculum-revision-2025/",
        "title": "NSBM Faculty of Computing – Curriculum Revision 2025 Announcement",
        "category": "curriculum_revision_notice",
        "authority": "NSBM Green University official announcement; informational notice, not the curriculum itself",
        "version": "Published 2025-12-18",
        "note": (
            "This announcement confirms a curriculum-revision exercise for Computing programmes, "
            "including BSc (Hons) Data Science. It does not publish the revised module handbook, "
            "module mapping, or approved curriculum document. Do not infer curriculum changes from this notice."
        ),
    },
    {
        "path": "examinations/NSBM_Examination_Portal_Public_Information.md",
        "url": "https://sites.google.com/view/nsbmexaminations",
        "title": "NSBM Examination Portal – Public Information",
        "category": "examination_information",
        "authority": "NSBM Examination Department official portal",
        "version": "Website snapshot " + TODAY,
        "note": (
            "The portal links to examination rules, EC submission, repeat coursework, payments "
            "and a live monthly examination schedule. The schedule is not copied as a static "
            "current schedule because it changes over time."
        ),
    },
    {
        "path": "examinations/NSBM_Extenuating_Circumstances_Online_Process_2026.md",
        "url": "https://sites.google.com/view/nsbmexaminations/submit-extenuating-circumstances-form-ec",
        "title": "NSBM Extenuating Circumstances – Online Submission Procedure",
        "category": "extenuating_circumstances",
        "authority": "NSBM Examination Department official procedure page",
        "version": "Effective from 2026-05-04; snapshot " + TODAY,
        "note": (
            "This is the newer online process page. The August 2023 EC PDF remains a separate "
            "older guideline. The page states the process applies to NSBM examinations, not "
            "partner-university examinations."
        ),
    },
    {
        "path": "student_services/NSBM_Student_Portal_Public_Directory.md",
        "url": "https://students.nsbm.ac.lk/",
        "title": "NSBM Student Portal – Public Service Directory",
        "category": "student_portal_directory",
        "authority": "NSBM Green University official Student Portal",
        "version": "Website snapshot " + TODAY,
        "note": "Public portal directory only. Login-protected student records and services are not captured.",
    },
    {
        "path": "student_services/NSBM_NFORCE_Career_Guidance_Unit.md",
        "url": "https://www.nsbm.ac.lk/career-guidance-unit/",
        "title": "NSBM NFORCE – Career and Entrepreneurship Unit",
        "category": "career_guidance",
        "authority": "NSBM Green University official NFORCE page",
        "version": "Website snapshot " + TODAY,
        "note": "General description of career guidance and industry-readiness services; individual opportunities may change.",
    },
    {
        "path": "student_services/NSBM_NFORCE_Portal_Public_Information.md",
        "url": "https://nforce.nsbm.ac.lk/student/index.php",
        "title": "NSBM NFORCE Portal – Public Information",
        "category": "career_guidance",
        "authority": "NSBM official NFORCE student portal",
        "version": "Website snapshot " + TODAY,
        "note": "Public landing-page snapshot only. Vacancies and portal services are dynamic and may require an NSBM login.",
    },
    {
        "path": "student_services/NSBM_Official_FAQ.md",
        "url": "https://www.nsbm.ac.lk/faq/",
        "title": "NSBM Green University – Official FAQ",
        "category": "official_faq",
        "authority": "NSBM Green University official FAQ; general information",
        "version": "Website snapshot " + TODAY,
        "note": "General FAQ. Programme-specific rules, fees and current academic procedures must be checked against their current official source.",
    },
    {
        "path": "research_and_projects/NSBM_Research_Council_Policies_and_Documents.md",
        "url": "https://www.nsbm.ac.lk/policies-and-documents/",
        "title": "NSBM Research Council – Policies and Documents",
        "category": "research_governance",
        "authority": "NSBM Research Council official page; primarily staff research-grant governance",
        "version": "Website snapshot " + TODAY,
        "note": "The public page lists research-grant administration forms and conditions. It is not a student thesis ethics policy.",
    },
    {
        "path": "research_and_projects/NSBM_Library_Research_Support.md",
        "url": "https://library.nsbm.ac.lk/pages/research-support.php",
        "title": "NSBM Library – Research Support and Thesis Guidelines",
        "category": "research_support",
        "authority": "NSBM Green University Library official research-support page",
        "version": "Website snapshot " + TODAY,
        "note": "Use together with the downloaded Thesis Preparation and Formatting Guidelines PDF.",
    },
    {
        "path": "student_services/NSBM_Library_Using_the_Library.md",
        "url": "https://library.nsbm.ac.lk/pages/using-the-library.php",
        "title": "NSBM Library – Using the Library and Service Policies",
        "category": "library_services",
        "authority": "NSBM Green University Library official website; service information",
        "version": "Website snapshot " + TODAY,
        "note": (
            "This is a dated website snapshot. Confirm current charges, borrowing limits, "
            "opening hours and clearance rules with the Library before giving a definitive answer."
        ),
    },
    {
        "path": "student_services/NSBM_Library_Learning_Support.md",
        "url": "https://library.nsbm.ac.lk/pages/learning-support.php",
        "title": "NSBM Library – Learning Support",
        "category": "learning_support",
        "authority": "NSBM Green University Library official service page",
        "version": "Website snapshot " + TODAY,
        "note": "General student learning-support service information.",
    },
    {
        "path": "student_services/NSBM_Library_Technical_Support.md",
        "url": "https://library.nsbm.ac.lk/pages/technical-support.php",
        "title": "NSBM Library – Technical Support",
        "category": "it_support",
        "authority": "NSBM Green University Library official service page",
        "version": "Website snapshot " + TODAY,
        "note": "General Library technical-support information; does not replace account-specific IT support.",
    },
]


SKIP_TAGS = {
    "script", "style", "noscript", "svg", "header", "footer", "nav",
    "form", "iframe", "button", "select", "option"
}
BLOCK_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr",
    "td", "th", "br", "section", "article", "main", "div"
}


class PlainTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth == 0:
            value = re.sub(r"\s+", " ", data).strip()
            if value:
                self.parts.append(value)


def fetch_bytes(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "")
    if not data:
        raise RuntimeError("Empty response")
    return data, content_type


def validate_pdf(data: bytes) -> int:
    if b"%PDF" not in data[:1024]:
        raise RuntimeError("Downloaded content is not a PDF")
    reader = PdfReader(BytesIO(data))
    if len(reader.pages) < 1:
        raise RuntimeError("PDF contains no pages")
    return len(reader.pages)


def clean_lines(text: str) -> str:
    raw_lines = re.split(r"[\r\n]+", text)
    result = []
    previous = None
    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line == previous:
            continue
        result.append(line)
        previous = line
    return "\n".join(result)


def remove_programme_fee_section(text: str) -> str:
    output = []
    removing = False
    for line in text.splitlines():
        low = line.casefold().strip()
        if "course fee" in low:
            removing = True
            continue
        if removing and "employment opportunities" in low:
            removing = False
        if not removing:
            output.append(line)
    return "\n".join(output)


def html_to_text(data: bytes) -> str:
    try:
        html = data.decode("utf-8")
    except UnicodeDecodeError:
        html = data.decode("latin-1", errors="replace")
    parser = PlainTextExtractor()
    parser.feed(html)
    parser.close()
    return clean_lines("\n".join(parser.parts))


def save_pdf_source(source: dict) -> dict:
    target = NSBM_ROOT / source["relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)

    page_count = None
    action = "existing"

    if target.exists():
        try:
            page_count = validate_pdf(target.read_bytes())
        except Exception:
            page_count = None

    if page_count is None:
        data, content_type = fetch_bytes(source["url"])
        page_count = validate_pdf(data)
        target.write_bytes(data)
        action = "downloaded"

    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    return {
        "relative_path": source["relative_path"].replace("\\", "/"),
        "document_id": "institutions/nsbm/" + str(Path(source["relative_path"]).with_suffix("")).replace("\\", "/"),
        "document_scope": "institution",
        "institution_id": "NSBM",
        "document_category": source["category"],
        "document_title": source["title"],
        "document_version": source["version"],
        "source_authority": source["authority"],
        "source_url": source["url"],
        "file_type": "pdf",
        "page_count": page_count,
        "sha256": digest,
        "collection_date": TODAY,
        "curation_note": source.get("note", ""),
        "action": action,
    }


def save_page_source(source: dict) -> dict:
    target = NSBM_ROOT / source["path"]
    target.parent.mkdir(parents=True, exist_ok=True)

    data, content_type = fetch_bytes(source["url"])
    text = html_to_text(data)

    if len(text) < 250:
        raise RuntimeError(
            f"Page text too short ({len(text)} chars); likely login/dynamic page"
        )

    if source.get("remove_programme_fee_section"):
        text = remove_programme_fee_section(text)

    if len(text) < 250:
        raise RuntimeError("Page text too short after curation")

    note = source.get("note", "")
    body = [
        f"# {source['title']}",
        "",
        f"- Source URL: {source['url']}",
        f"- Retrieved on: {TODAY}",
        f"- Source authority: {source['authority']}",
        f"- Category: {source['category']}",
        f"- Version: {source['version']}",
    ]
    if note:
        body += ["", "## Source and scope note", note]
    body += ["", "## Official page text snapshot", "", text, ""]

    rendered = "\n".join(body)
    target.write_text(rendered, encoding="utf-8")

    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    return {
        "relative_path": source["path"].replace("\\", "/"),
        "document_id": "institutions/nsbm/" + str(Path(source["path"]).with_suffix("")).replace("\\", "/"),
        "document_scope": "institution",
        "institution_id": "NSBM",
        "document_category": source["category"],
        "document_title": source["title"],
        "document_version": source["version"],
        "source_authority": source["authority"],
        "source_url": source["url"],
        "file_type": "markdown",
        "character_count": len(text),
        "sha256": digest,
        "collection_date": TODAY,
        "curation_note": note,
        "action": "snapshot_saved",
    }


def main():
    NSBM_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("NSBM OFFICIAL SOURCE COLLECTION")
    print("=" * 68)
    print("Project:", ROOT)
    print("NSBM root:", NSBM_ROOT)
    print("Collection date:", TODAY)

    records = []
    failures = []

    print("\n[1/2] Validating/downloading official PDFs")
    for source in PDF_SOURCES:
        try:
            record = save_pdf_source(source)
            records.append(record)
            print(
                f"PASS | PDF | {record['action']} | "
                f"{record['document_title']} | "
                f"{record['page_count']} pages"
            )
        except Exception as exc:
            failures.append((source["filename"], str(exc)))
            print(f"FAIL | PDF | {source['filename']} | {type(exc).__name__}: {exc}")

    print("\n[2/2] Saving official website snapshots")
    for source in PAGE_SOURCES:
        try:
            record = save_page_source(source)
            records.append(record)
            print(
                f"PASS | PAGE | {record['character_count']} chars | "
                f"{record['document_title']}"
            )
        except Exception as exc:
            failures.append((source["url"], str(exc)))
            print(f"FAIL | PAGE | {source['url']} | {type(exc).__name__}: {exc}")

    # Preserve previous manifest entries that are not part of this collection run.
    previous = []
    if MANIFEST_PATH.exists():
        try:
            old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            previous = old.get("documents", [])
        except Exception:
            print("WARNING: Existing manifest could not be read; a new manifest will be written.")

    by_path = {
        item.get("relative_path"): item
        for item in previous
        if item.get("relative_path")
    }
    for item in records:
        by_path[item["relative_path"]] = item

    manifest = {
        "institution_id": "NSBM",
        "institution_name": "NSBM Green University",
        "collection_date": TODAY,
        "root_directory": "data/documents/institutions/nsbm",
        "documents": sorted(
            by_path.values(),
            key=lambda item: item.get("relative_path", "")
        ),
        "collection_failures": [
            {"source": source, "error": error}
            for source, error in failures
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("COLLECTION SUMMARY")
    print("=" * 68)
    counts = Counter(item["file_type"] for item in records)
    print("New/validated PDF sources:", counts.get("pdf", 0))
    print("Website text snapshots:", counts.get("markdown", 0))
    print("Manifest records:", len(manifest["documents"]))
    print("Failures:", len(failures))
    print("Manifest:", MANIFEST_PATH)

    if failures:
        print("\nSome sources could not be collected. Do not ingest yet.")
        sys.exit(1)

    print("\nCOLLECTION CHECK: PASSED")
    print("No Chroma database changes were made.")
    print("Existing NSBM root-level PDF files were not moved or deleted.")


if __name__ == "__main__":
    main()
