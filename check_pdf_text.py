
from pathlib import Path
from pypdf import PdfReader

files = list(Path("data/documents").rglob("*.pdf"))

print("TOTAL PDF FILES:", len(files))

for pdf in files:
    try:
        reader = PdfReader(str(pdf))
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        print("\nFILE:", pdf.name)
        print("PAGES:", len(reader.pages))
        print("EXTRACTED CHARACTERS:", len(text))
        print("SAMPLE:", repr(text[:250]))

        if len(text) > 100:
            print("STATUS: TEXT AVAILABLE")
        else:
            print("STATUS: CHECK PDF / SCANNED DOCUMENT")

    except Exception as error:
        print("ERROR:", pdf.name, error)
