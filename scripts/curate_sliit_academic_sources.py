from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIIT_ROOT = ROOT / "data" / "documents" / "institutions" / "sliit"
MANIFEST = SLIIT_ROOT / "sliit_sources_manifest.json"
BACKUPS = ROOT / "data" / "backups"

SOURCES = [
    {
        "title": "SLIIT General Timelines for Semester and Module Registration and Examinations (10 June 2025)",
        "category": "examinations",
        "filename": "SLIIT_General_Semester_Registration_and_Examination_Timelines_2025-06-10.md",
        "source_url": "https://support.sliit.lk/kb/articles/pdf/general-timelines-for-semester-module-registration-and-examination",
        "index": True,
        "body": """# SLIIT General Timelines for Semester and Module Registration and Examinations\n\n**Source publication date:** 10 June 2025  \n**Source authority:** SLIIT Academic Affairs / SLIIT Support Desk  \n**Source URL:** https://support.sliit.lk/kb/articles/pdf/general-timelines-for-semester-module-registration-and-examination\n\n## Important status and limitations\n\nThis is a general guideline published by SLIIT on 10 June 2025. The source explicitly says the timelines are subject to change based on institutional requirements and other specific situations. It also says CourseWeb is the primary communication platform for official student information and that registration and examination timelines are announced through CourseWeb notices. This document is not a current individual student's timetable or a guarantee of dates for a particular intake.\n\n## General semester timelines stated by SLIIT\n\n| Activity | General timing | January–June semester | July–December semester |\n|---|---|---|---|\n| Semester registration | Four weeks before semester commencement | December / January | May / June |\n| Late semester registration (penalty fees apply) | One week into semester commencement | January | June |\n| Semester commencement | — | February | July |\n| Elective registration | Two weeks after semester commencement | February | July |\n| Module registration | Three weeks after semester commencement at the latest | March | August |\n| Mid examination | Eight weeks after semester commencement | April | September |\n| Final examination | Fifteen weeks after semester commencement | May / June | October / November |\n| Provisional results release | — | July | December |\n\n## Guidance for answering student questions\n\n- Treat the table as a general planning guide only, not as a confirmed date list for a current semester.\n- For exact registration deadlines, examination dates, timetable changes, or result-release dates, direct students to the latest official CourseWeb notices and their relevant faculty or Academic Affairs office.\n- Do not infer a specific date from the week number or month shown in this general guide.\n- Where a programme is awarded by a partner university, check whether the partner university's rules and schedule apply.\n\n**Captured as a project Markdown summary of the public SLIIT Support Desk article; consult the source URL for the source document and updates.**\n""",
    },
    {
        "title": "SLIIT Student Rules and Regulations – Public Overview (24 January 2021)",
        "category": "academic_regulations",
        "filename": "SLIIT_Student_Rules_and_Regulations_Public_Overview_2021-01-24.md",
        "source_url": "https://support.sliit.lk/kb/articles/pdf/rules-and-regulations-for-students",
        "index": True,
        "body": """# SLIIT Student Rules and Regulations – Public Overview\n\n**Source publication date:** 24 January 2021  \n**Source authority:** SLIIT Academic Affairs (Registry) / SLIIT Support Desk  \n**Source URL:** https://support.sliit.lk/kb/articles/pdf/rules-and-regulations-for-students\n\n## Student Rule Book\n\nSLIIT's public Support Desk article explains that the Student Disciplinary Rules and Academic Administrative Regulations are commonly referred to as the Student Rule Book. The article identifies the **10th Edition – 2018** and states that it is available through CourseWeb at: Support Services → Student Affairs → Institute Manuals → Student Rule Book.\n\n## Rules that may apply to students\n\nThe article states that SLIIT students are bound by applicable Student Disciplinary Rules and Academic Administrative Regulations, including admission and registration regulations, as well as applicable rules for library, laboratories, hostel use (where relevant), IT policy, and other SLIIT regulations.\n\n## SLIIT programmes and partner-university programmes\n\nThe article distinguishes examination regulations as follows:\n\n- Students following diplomas and degrees offered by SLIIT are subject to SLIIT Examination Regulations.\n- Students following programmes leading to degrees offered by SLIIT partner universities are subject to the examination regulations of the applicable partner university.\n\n## Important version and access limitations\n\n- The public article is dated 24 January 2021 and refers to the Student Rule Book 10th Edition – 2018. It does not reproduce the complete Rule Book.\n- Do not assume this is the latest edition or that every provision remains unchanged. Students should consult the current Rule Book and notices through their official CourseWeb access.\n- Do not apply SLIIT examination rules to a partner-university award without first identifying the awarding university and applicable regulations.\n\n**Captured as a project Markdown summary of the public SLIIT Support Desk article; it is not a replacement for the full Student Rule Book.**\n""",
    },
    {
        "title": "SLIIT Notice on Reference Sheets for Final Examinations (May 2025)",
        "category": "examinations",
        "filename": "SLIIT_Reference_Sheets_for_Final_Examinations_Notice_2025-05.md",
        "source_url": "https://courseweb.sliit.lk/pluginfile.php/113/mod_forum/attachment/10191/Introduction%20of%20Reference%20Sheets%20to%20be%20used%20during%20Examinations.pdf?forcedownload=1",
        "index": True,
        "body": """# SLIIT Notice: Reference Sheets to Be Used During Examinations\n\n**Source date:** May 2025  \n**Source authority:** Director, Academic Affairs and Student Services, SLIIT  \n**Source URL:** https://courseweb.sliit.lk/pluginfile.php/113/mod_forum/attachment/10191/Introduction%20of%20Reference%20Sheets%20to%20be%20used%20during%20Examinations.pdf?forcedownload=1\n\n## When a reference sheet may be used\n\nThe notice says that, following a Senate recommendation, faculties may permit one self-created A4-size reference sheet in final examinations **at the discretion of the Lecturer-in-Charge (LIC)**. It is therefore not a universal entitlement for every module or examination. Students should follow the instructions given by the LIC for the relevant module.\n\n## Requirements stated in the notice\n\n- The reference sheet is limited to one A4 page, handwritten or printed, and may use one or both sides.\n- It should be concise and contain only material taught during the module.\n- Sample answers must not be included.\n- Students must not share reference sheets during examinations; each student prepares and uses their own sheet.\n- The module name and examination candidate's index number must appear at the top.\n- The reference sheet must be attached to the answer script.\n- The LIC may provide further module-specific instructions.\n\n## Academic misconduct warning\n\nThe notice refers to an update to Section 2.8 of the SLIIT Student Rule Book. Possession of unauthorised notes or materials and copying/attempting to copy are identified as examination misconduct. The notice says consequences can include failing the examination, suspension, or expulsion.\n\n## Important limitations\n\nThis summary represents the May 2025 notice only. It must not be treated as confirmation that reference sheets are permitted in every current examination. Students should check the latest CourseWeb notice and obtain module-specific confirmation from the LIC.\n\n**Captured as a project Markdown summary of a publicly accessible SLIIT CourseWeb notice PDF.**\n""",
    },
    {
        "title": "SLIIT Public CourseWeb Calendar – Access and Snapshot Limitations (14 November 2026)",
        "category": "academic_calendars",
        "filename": "SLIIT_CourseWeb_Public_Calendar_Access_and_Limitations_2026-11-14.md",
        "source_url": "https://courseweb.sliit.lk/course/index.php?categoryid=2117",
        "index": False,
        "body": """# SLIIT Public CourseWeb Calendar – Access and Snapshot Limitations\n\n**Checked:** 14 November 2026  \n**Source authority:** SLIIT CourseWeb  \n**Source URL:** https://courseweb.sliit.lk/course/index.php?categoryid=2117\n\n## What is publicly visible\n\nThe public CourseWeb page exposes a Calendar block and semester category links. The public view checked on 14 November 2026 displayed November 2026 and month navigation to October and December. The calendar view showed event counts on selected days, but the public page did not provide a complete official academic calendar document or a full set of event details in the accessible snapshot. Course content and student-specific notices require SLIIT login.\n\n## How this source should be used\n\nThis is an **archive/access note only**, not an academic calendar and not an examination timetable. It is deliberately marked `index: false` in the SLIIT manifest so that the limited public calendar snapshot is not used as answer evidence.\n\nFor exact semester dates, examination dates, registration deadlines, holidays, or timetable changes, students must consult the latest official CourseWeb notices and the relevant faculty/Academic Affairs communications. Do not infer dates from this snapshot.\n""",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        raise SystemExit(f"SLIIT manifest not found: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("institution_id") != "SLIIT":
        raise SystemExit("Manifest institution_id is not SLIIT; no changes made.")

    records = manifest.get("sources")
    if not isinstance(records, list):
        raise SystemExit("Manifest sources is not a list; no changes made.")

    existing_urls = {r.get("source_url") for r in records if isinstance(r, dict)}
    existing_paths = {str(r.get("local_path", "")).replace("\\", "/") for r in records if isinstance(r, dict)}
    new_records = []
    skipped = 0
    now = datetime.now(timezone.utc).isoformat()

    for item in SOURCES:
        rel = f"data/documents/institutions/sliit/{item['category']}/{item['filename']}"
        destination = ROOT / Path(rel)
        if item["source_url"] in existing_urls or rel in existing_paths:
            skipped += 1
            print(f"[SKIP EXISTING] {item['filename']}")
            continue
        if destination.exists():
            raise SystemExit(f"Destination already exists but is not registered: {destination}. No overwrite performed.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = item["body"].strip() + "\n"
        destination.write_text(content, encoding="utf-8", newline="\n")
        record = {
            "title": item["title"],
            "category": item["category"],
            "format": "markdown_snapshot",
            "source_type": "curated_summary_of_public_official_source",
            "source_url": item["source_url"],
            "local_path": rel,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "status": "downloaded",
            "index": item["index"],
            "collected_at_utc": now,
        }
        if not item["index"]:
            record["index_note"] = "Archive/access note only; not a formal calendar or answer evidence."
        new_records.append(record)
        print(f"[ADDED] {item['category']} | {item['filename']} | index={item['index']}")
        existing_urls.add(item["source_url"])
        existing_paths.add(rel)

    if not new_records:
        print("No new source summaries were needed.")
        print("Manifest unchanged.")
        return 0

    BACKUPS.mkdir(parents=True, exist_ok=True)
    backup = BACKUPS / f"sliit_manifest_before_academic_source_curation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(MANIFEST, backup)

    manifest["sources"] = records + new_records
    manifest["source_count"] = len(manifest["sources"])
    manifest["pdf_count"] = sum(1 for r in manifest["sources"] if str(r.get("local_path", "")).lower().endswith(".pdf"))
    manifest["page_snapshot_count"] = manifest["source_count"] - manifest["pdf_count"]
    manifest["new_sources_this_run"] = len(new_records)
    manifest["existing_sources_skipped_this_run"] = skipped
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    temp_path = MANIFEST.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temp_path.replace(MANIFEST)

    print("=" * 68)
    print("SLIIT ACADEMIC SOURCE CURATION COMPLETE")
    print("New curated source summaries:", len(new_records))
    print("Archive-only sources added:", sum(1 for r in new_records if r.get("index") is False))
    print("Manifest sources:", manifest["source_count"])
    print("PDFs:", manifest["pdf_count"])
    print("Text snapshots:", manifest["page_snapshot_count"])
    print("Manifest:", MANIFEST.relative_to(ROOT))
    print("Backup:", backup.relative_to(ROOT))
    print("No existing source files were overwritten or deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
