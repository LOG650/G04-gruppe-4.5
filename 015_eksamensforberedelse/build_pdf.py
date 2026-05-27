"""Build PDF from exam preparation markdown via Chrome headless.

Usage:
    python build_pdf.py
"""
from pathlib import Path
import markdown, subprocess

HERE = Path(__file__).parent
MD_FILE = HERE / "Eksamensforberedelse_muntlig.md"
HTML_FILE = HERE / "_tmp_eksamen.html"
PDF_FILE = HERE / "Eksamensforberedelse_muntlig.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

md_text = MD_FILE.read_text(encoding="utf-8")

extensions = ["tables", "fenced_code", "smarty"]
html_body = markdown.markdown(md_text, extensions=extensions)

full_html = f"""<!DOCTYPE html>
<html lang="nb">
<head>
<meta charset="utf-8">
<title>Eksamensforberedelse — muntlig eksamen LOG650</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '$$', right: '$$', display: true}},
    {{left: '$', right: '$', display: false}}
  ]}});"></script>
<style>
@page {{
  size: A4;
  margin: 22mm 18mm 18mm 18mm;
}}
body {{
  font-family: 'Segoe UI', Calibri, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.6;
  color: #1f2937;
  max-width: 720px;
  margin: 0 auto;
  padding: 20px 0;
}}
h1 {{
  font-size: 18pt;
  color: #111827;
  border-bottom: 3px solid #c8102e;
  padding-bottom: 6px;
  margin-top: 32px;
  page-break-after: avoid;
}}
h2 {{
  font-size: 13pt;
  color: #1e40af;
  margin-top: 24px;
  page-break-after: avoid;
}}
h3 {{
  font-size: 11pt;
  color: #374151;
  margin-top: 18px;
  page-break-after: avoid;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0;
  font-size: 9.5pt;
}}
th, td {{
  border: 1px solid #d1d5db;
  padding: 6px 10px;
  text-align: left;
}}
th {{
  background: #f3f4f6;
  font-weight: 600;
}}
tr:nth-child(even) {{
  background: #f9fafb;
}}
blockquote {{
  border-left: 4px solid #c8102e;
  margin: 16px 0;
  padding: 10px 16px;
  background: #fef2f2;
  color: #374151;
}}
code {{
  background: #f3f4f6;
  padding: 2px 5px;
  border-radius: 3px;
  font-size: 9.5pt;
}}
pre {{
  background: #f3f4f6;
  padding: 12px;
  border-radius: 5px;
  overflow-x: auto;
  font-size: 9pt;
}}
hr {{
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 24px 0;
}}
strong {{ color: #111827; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

HTML_FILE.write_text(full_html, encoding="utf-8")

subprocess.run([
    CHROME,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    f"--print-to-pdf={PDF_FILE}",
    "--print-to-pdf-no-header",
    str(HTML_FILE.resolve().as_uri()),
], check=True)

HTML_FILE.unlink(missing_ok=True)
print(f"PDF written: {PDF_FILE.name} ({PDF_FILE.stat().st_size:,} bytes)")
