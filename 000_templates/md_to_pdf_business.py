"""Business-style markdown -> PDF generator with embedded logos.

Usage:
    python md_to_pdf_business.py <input.md> [--title "Document title"] [--subtitle "Subtitle"]
"""
import argparse
import base64
import markdown
import mimetypes
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOGOS_DIR = SCRIPT_DIR / "logos"
HIMOLDE_LOGO = LOGOS_DIR / "himolde_logo.png"
SKORINGEN_LOGO = LOGOS_DIR / "skoringen_logo.svg"


def embed_image(path: Path) -> str:
    """Return data: URI for an image file."""
    if not path.exists():
        return ""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path")
    parser.add_argument("--title", default="Møtereferat")
    parser.add_argument("--subtitle", default="Gruppe 4.5 — LOG650 Forskningsprosjekt")
    parser.add_argument("--doc-id", default="")
    parser.add_argument("--date-str", default="")
    args = parser.parse_args()

    md_path = Path(args.md_path)
    pdf_path = md_path.with_suffix(".pdf")
    html_path = md_path.with_suffix(".html")

    md_text = md_path.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"]
    )

    himolde_uri = embed_image(HIMOLDE_LOGO)
    skoringen_uri = embed_image(SKORINGEN_LOGO)

    css = """
@page {
  size: A4;
  margin: 30mm 20mm 22mm 20mm;
  @top-left { content: ""; }
  @top-right { content: ""; }
  @bottom-center {
    content: "Side " counter(page) " av " counter(pages);
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 8pt;
    color: #94a3b8;
  }
}
@page :first {
  margin-top: 0mm;
}

* { box-sizing: border-box; }
html, body {
  font-family: 'Inter', 'Segoe UI', 'Calibri', Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1f2937;
  margin: 0;
  padding: 0;
}

/* ===== LETTERHEAD ===== */
.letterhead {
  background: #ffffff;
  padding: 14mm 20mm 12mm 20mm;
  border-bottom: 4px solid #c8102e;
  display: grid;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  gap: 20mm;
  margin-bottom: 0;
}
.letterhead-left img {
  max-height: 16mm;
  width: auto;
}
.letterhead-right {
  text-align: right;
}
.letterhead-right img {
  max-height: 14mm;
  width: auto;
  filter: brightness(0.15);
}
.letterhead-strip {
  display: flex;
  height: 2mm;
  margin-bottom: 0;
}
.letterhead-strip > div { flex: 1; }
.strip-1 { background: #c8102e; }
.strip-2 { background: #1e3a5f; }
.strip-3 { background: #f4f6f8; }

/* ===== DOCUMENT TITLE BLOCK ===== */
.title-block {
  padding: 12mm 20mm 8mm 20mm;
  background: #fafbfc;
  border-bottom: 1px solid #e2e8f0;
  margin-bottom: 8mm;
}
.title-eyebrow {
  font-size: 8.5pt;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: #c8102e;
  font-weight: 600;
  margin-bottom: 3mm;
}
.title-main {
  font-size: 24pt;
  font-weight: 700;
  letter-spacing: -0.015em;
  color: #1e3a5f;
  line-height: 1.1;
  margin-bottom: 3mm;
}
.title-sub {
  font-size: 12pt;
  color: #475569;
  font-weight: 400;
  margin-bottom: 6mm;
}
.title-meta {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6mm;
  padding-top: 5mm;
  border-top: 1px solid #e2e8f0;
}
.meta-item .meta-label {
  font-size: 7.5pt;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #94a3b8;
  font-weight: 600;
  margin-bottom: 1mm;
}
.meta-item .meta-value {
  font-size: 10.5pt;
  color: #1e3a5f;
  font-weight: 600;
}

/* ===== CONTENT ===== */
.content {
  padding: 0 0 0 0;
}
h1 {
  font-size: 16pt;
  color: #1e3a5f;
  font-weight: 700;
  margin: 9mm 0 4mm 0;
  padding-bottom: 2mm;
  border-bottom: 2px solid #c8102e;
  letter-spacing: -0.01em;
}
h1:first-of-type { margin-top: 0; }
h2 {
  font-size: 13pt;
  color: #1e3a5f;
  font-weight: 600;
  margin: 7mm 0 2mm 0;
  padding-left: 3mm;
  border-left: 4px solid #c8102e;
  letter-spacing: -0.005em;
}
h3 {
  font-size: 11.5pt;
  color: #1e3a5f;
  font-weight: 600;
  margin: 5mm 0 2mm 0;
}
h4 {
  font-size: 10.5pt;
  color: #334155;
  font-weight: 600;
  margin: 4mm 0 1mm 0;
}
p {
  margin: 2mm 0;
  text-align: justify;
  hyphens: auto;
}
ul, ol {
  margin: 2mm 0 3mm 0;
  padding-left: 7mm;
}
li {
  margin: 1.5mm 0;
}
strong {
  color: #1e3a5f;
  font-weight: 600;
}
em { color: #334155; }
hr {
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 6mm 0;
}
blockquote {
  border-left: 4px solid #c8102e;
  margin: 4mm 0;
  padding: 3mm 5mm;
  background: #fef2f4;
  color: #1e3a5f;
  border-radius: 0 2mm 2mm 0;
  font-style: italic;
  font-size: 10pt;
}
blockquote p { margin: 1mm 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 4mm 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #cbd5e1;
  padding: 2.5mm 3mm;
  text-align: left;
  vertical-align: top;
}
th {
  background: #1e3a5f;
  color: #ffffff;
  font-weight: 600;
  letter-spacing: 0.02em;
}
tr:nth-child(even) td { background: #f8fafc; }
code {
  background: #f1f5f9;
  padding: 0.5mm 1.5mm;
  border-radius: 1mm;
  font-family: 'JetBrains Mono', Consolas, monospace;
  font-size: 9pt;
  color: #c8102e;
}
pre {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 2mm;
  padding: 3mm 4mm;
  overflow-x: auto;
  font-size: 9pt;
}
pre code {
  background: transparent;
  padding: 0;
  color: #1e293b;
}

/* ===== SIGN-OFF ===== */
.signoff {
  margin-top: 10mm;
  padding: 5mm 6mm;
  background: #1e3a5f;
  color: #ffffff;
  border-radius: 2mm;
  font-size: 10pt;
}
.signoff strong { color: #ffffff; }

/* ===== UTILITY ===== */
.page-break { page-break-after: always; }
.no-break { page-break-inside: avoid; }
"""

    himolde_img = (
        f'<img src="{himolde_uri}" alt="Høgskolen i Molde">' if himolde_uri else ""
    )
    skoringen_img = (
        f'<img src="{skoringen_uri}" alt="Skoringen">' if skoringen_uri else ""
    )

    letterhead = f"""
<div class="letterhead">
  <div class="letterhead-left">{himolde_img}</div>
  <div class="letterhead-right">{skoringen_img}</div>
</div>
<div class="letterhead-strip">
  <div class="strip-1"></div>
  <div class="strip-2"></div>
  <div class="strip-3"></div>
</div>
"""

    title_block = f"""
<div class="title-block">
  <div class="title-eyebrow">LOG650 &middot; Forskningsprosjekt i logistikk &middot; Vårsemesteret 2026</div>
  <div class="title-main">{args.title}</div>
  <div class="title-sub">{args.subtitle}</div>
  <div class="title-meta">
    <div class="meta-item">
      <div class="meta-label">Dokument-ID</div>
      <div class="meta-value">{args.doc_id or md_path.stem}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Dato</div>
      <div class="meta-value">{args.date_str or '14. mai 2026'}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Gruppe</div>
      <div class="meta-value">G04 &middot; 4.5</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Casebedrift</div>
      <div class="meta-value">Skoringen Råholt</div>
    </div>
  </div>
</div>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="nb">
<head>
<meta charset="utf-8">
<title>{args.title}</title>
<style>{css}</style>
</head>
<body>
{letterhead}
{title_block}
<div class="content">
{body_html}
</div>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge = next((p for p in edge_paths if os.path.exists(p)), None)
    if edge is None:
        print("Edge not found", file=sys.stderr)
        sys.exit(1)

    file_url = "file:///" + str(html_path).replace("\\", "/")
    cmd = [
        edge,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        file_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    print("returncode:", result.returncode)
    print("PDF created at:", pdf_path, "exists:", pdf_path.exists())
    if result.stderr:
        print("stderr (info only):", result.stderr[:200])


if __name__ == "__main__":
    main()
