from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("data/documents/raw")

documents = {
    "TCS Quarterly Results": {
        "filename": "TCS_results_2026-10-08.pdf",
        "keywords": ["revenue", "profit", "quarter"],
    },
    "TCS Earnings Call Transcript": {
        "filename": "TCS_transcript_2026-10-08.pdf",
        "keywords": ["revenue", "growth", "question"],
    },
    "TCS Annual Report": {
        "filename": "TCS_annual_report_2026-05-15.pdf",
        "keywords": ["annual report", "revenue", "risk"],
    },
    "Infosys Quarterly Results": {
        "filename": "INFY_results_2026-07-23.pdf",
        "keywords": ["revenue", "operating margin", "profit"],
    },
    "Infosys Earnings Call Transcript": {
        "filename": "INFY_transcript_2026-07-23.pdf",
        "keywords": ["earnings conference call", "revenue", "growth"],
    },
    "Infosys Annual Report": {
        "filename": "INFY_annual_report_2026-05-29.pdf",
        "keywords": ["annual report", "financial statements", "revenue"],
    },
}

passed = 0
failed = 0

for document_type, details in documents.items():
    pdf_path = PDF_DIR / details["filename"]

    print("\n" + "=" * 65)
    print(f"DOCUMENT: {document_type}")
    print(f"FILE: {pdf_path.name}")
    print("=" * 65)

    if not pdf_path.is_file():
        print("FAIL: PDF file not found.")
        failed += 1
        continue

    try:
        reader = PdfReader(str(pdf_path))

        extracted_pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]

        full_text = "\n".join(extracted_pages)

        print(f"Pages: {len(reader.pages)}")
        print(f"Extracted characters: {len(full_text)}")

        if not full_text.strip():
            print("FAIL: No extractable text.")
            failed += 1
            continue

        print("PASS: Text extraction succeeded.")

        print("\nKeyword checks:")

        for keyword in details["keywords"]:
            found = keyword.lower() in full_text.lower()
            print(f"  {keyword}: {'FOUND' if found else 'NOT FOUND'}")

        print("\nText sample:")
        print(full_text[:500])

        passed += 1

    except Exception as error:
        print(f"FAIL: {error}")
        failed += 1

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"Documents passed: {passed}")
print(f"Documents failed: {failed}")
print(f"Total documents: {len(documents)}")