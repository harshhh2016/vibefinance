# These three files are standalone INFY ingestion scripts (they exit at import),
# not tests, so pytest must not collect them.
collect_ignore = [
    "test_infy_pdfs.py",
    "test_infy_financial_content.py",
    "test_all_financial_pdfs.py",
]
