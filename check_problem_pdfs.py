from pathlib import Path
import fitz

folders = [
    Path("data/documents/institutions/university_of_colombo"),
    Path("data/documents/institutions/university_of_moratuwa"),
]

for folder in folders:
    for pdf in folder.glob("*.pdf"):
        print("\n" + "=" * 60)
        print("FILE:", pdf.name)

        try:
            doc = fitz.open(pdf)
            total_chars = 0

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                total_chars += len(text)

                if page_num == 1:
                    print("FIRST PAGE SAMPLE:")
                    print(text[:500])

            print("TOTAL PAGES:", len(doc))
            print("TOTAL EXTRACTED CHARACTERS:", total_chars)

            if total_chars < 500:
                print("STATUS: NEED OCR / TEXT EXTRACTION CHECK")
            else:
                print("STATUS: TEXT EXTRACTION AVAILABLE")

        except Exception as e:
            print("ERROR:", e)