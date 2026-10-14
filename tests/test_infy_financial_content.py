from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("data/documents/raw")

documents = {
    "Quarterly Results": {
        "filename": "INFY_results_2026-07-23.pdf",
        "keywords": [
            "revenue",
            "operating margin",
            "profit",
            "Q1",
        ],
    },
    "Earnings Call Transcript": {
        "filename": "INFY_transcript_2026-07-23.pdf",
        "keywords": [
            "earnings conference call",
            "revenue",
            "growth",
            "question",
            "answer",
        ],
    },
    "Annual Report": {
        "filename": "INFY_annual_report_2026-05-29.pdf",
        "keywords": [
            "annual report",
            "financial statements",
            "revenue",
            "risk",
            "profit",
        ],
    },
}

for document_type, details in documents.items():
    pdf_path = PDF_DIR / details["filename"]

    print("\n" + "=" * 65)
    print(f"DOCUMENT TYPE: {document_type}")
    print(f"FILENAME: {pdf_path.name}")
    print("=" * 65)

    # Check whether the PDF exists
    if not pdf_path.is_file():
        print("FAIL: PDF file not found.")
        print(f"Expected path: {pdf_path.resolve()}")
        continue

    try:
        # Open the PDF
        reader = PdfReader(str(pdf_path))

        print(f"Pages: {len(reader.pages)}")

        # Extract text from all pages
        extracted_pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            extracted_pages.append(text)

        full_text = "\n".join(extracted_pages)

        print(f"Extracted characters: {len(full_text)}")

        # Check whether text was extracted
        if not full_text.strip():
            print("FAIL: No text could be extracted.")
            continue

        print("PASS: Text extraction succeeded.")

        # Check for relevant financial keywords
        print("\nKeyword checks:")

        for keyword in details["keywords"]:
            found = keyword.lower() in full_text.lower()

            status = "FOUND" if found else "NOT FOUND"

            print(f"  {keyword}: {status}")

        # Display a sample of extracted text
        print("\n--- Extracted Text Sample (first 1000 characters) ---\n")

        print(full_text[:1000])

    except Exception as error:
        print(f"FAIL: Could not process PDF: {error}")

print("\n" + "=" * 65)
print("PDF CONTENT TESTING COMPLETED")
print("=" * 65)