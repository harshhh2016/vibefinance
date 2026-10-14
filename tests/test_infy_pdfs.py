from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("data/documents/raw")

pdf_files = sorted(PDF_DIR.glob("INFY*.pdf"))

if not pdf_files:
    print("ERROR: No Infosys PDFs found.")
    raise SystemExit(1)

for pdf_path in pdf_files:
    print("\n" + "=" * 60)
    print(f"FILE: {pdf_path.name}")
    print("=" * 60)

    try:
        reader = PdfReader(str(pdf_path))

        print(f"Number of pages: {len(reader.pages)}")

        extracted_text = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            extracted_text.append(text)

        full_text = "\n".join(extracted_text)

        print(f"Extracted characters: {len(full_text)}")

        if not full_text.strip():
            print("FAIL: No text could be extracted.")
            continue

        print("PASS: Text extraction succeeded.")

        print("\n--- First 500 characters ---\n")
        print(full_text[:500])

        print("\n--- Financial keyword check ---")

        keywords = [
            "revenue",
            "profit",
            "quarter",
            "financial",
            "growth",
        ]

        text_lower = full_text.lower()

        for keyword in keywords:
            found = keyword in text_lower
            print(f"{keyword}: {'FOUND' if found else 'NOT FOUND'}")

    except Exception as error:
        print(f"FAIL: {error}")