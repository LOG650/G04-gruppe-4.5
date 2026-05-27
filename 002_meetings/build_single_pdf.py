"""Build PDF from a single markdown file using Chrome headless."""
import markdown
from pathlib import Path
import subprocess, sys

HERE = Path(__file__).parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
@page { size: A4; margin: 22mm 18mm 18mm 18mm; }
body { font-family: 'Segoe UI', Calibri, sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1f2937; max-width: 720px; margin: 0 auto; padding: 20px 0; }
h1 { font-size: 18pt; color: #111827; border-bottom: 3px solid #c8102e; padding-bottom: 6px; }
h2 { font-size: 13pt; color: #1e40af; margin-top: 22px; }
h3 { font-size: 11pt; color: #374151; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th, td { border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; }
th { background: #f3f4f6; font-weight: 600; }
tr:nth-child(even) { background: #f9fafb; }
blockquote { border-left: 4px solid #c8102e; margin: 16px 0; padding: 10px 16px; background: #fef2f2; }
code { background: #f3f4f6; padding: 2px 5px; border-radius: 3px; font-size: 9.5pt; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }
strong { color: #111827; }
"""

def build(md_path):
    md_path = Path(md_path)
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "smarty"])

    tmp_html = md_path.with_suffix(".tmp.html")
    pdf_path = md_path.with_suffix(".pdf")

    full_html = f"<!DOCTYPE html><html lang='nb'><head><meta charset='utf-8'><title>{md_path.stem}</title><style>{CSS}</style></head><body>{html_body}</body></html>"
    tmp_html.write_text(full_html, encoding="utf-8")

    url = tmp_html.resolve().as_uri()
    subprocess.run([
        CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}", url
    ], capture_output=True, check=True)

    tmp_html.unlink(missing_ok=True)
    print(f"PDF: {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python build_single_pdf.py <file.md>")
    for arg in sys.argv[1:]:
        build(HERE / arg)
