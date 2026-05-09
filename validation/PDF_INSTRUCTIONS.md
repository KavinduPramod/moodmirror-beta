# Creating 10 Clean Review PDFs

This project already has 10 structured case files in:

- `validation/review_sample/user_01.json` … `user_10.json`

Use the exporter script to convert each JSON file into a readable PDF.

## 1) Install dependency

From workspace root:

```bash
pip install reportlab
```

## 2) Generate PDFs

From workspace root:

```bash
python validation/export_review_pdfs.py
```

Output folder (default):

- `validation/review_sample_pdf/`
- Creates: `user_01.pdf` … `user_10.pdf`

## 3) If you want blinded PDFs (recommended for independent review)

This hides the `automated_result` section:

```bash
python validation/export_review_pdfs.py --hide-automated
```

## 4) If you want only user data (cleanest output)

This keeps only:
- case ID
- behavioral summary
- sample posts

```bash
python validation/export_review_pdfs.py --user-only
```

## 5) Optional custom input/output folders

```bash
python validation/export_review_pdfs.py --input-dir validation/review_sample --output-dir validation/review_sample_pdf
```

## Notes for proper PDF quality

- One case per PDF for easy sharing and printing.
- Structured sections: summary, posts, expert classification.
- Consistent typography and spacing for readability.
- UTF-8 support keeps text from Reddit readable.