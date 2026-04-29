"""Build .html + .pdf next to a given oppsummering markdown file.

Usage:
    python build_oppsummering.py 2026-04-29_oppsummering_til_gruppa.md

Wraps the markdown content in the same self-contained HTML template that
2026-04-26_sondag.html uses (marked.js + responsive print CSS), then runs
Chrome headless to produce a matching PDF.
"""
from pathlib import Path
import re, subprocess, sys

HERE = Path(__file__).parent
TEMPLATE = HERE / "2026-04-26_sondag.html"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def build(md_path: Path) -> None:
    if not md_path.is_file():
        sys.exit(f"Not found: {md_path}")

    md = md_path.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")

    # Extract the title from the first H1 line and any explicit "Dato:" field
    title_line = next((ln.lstrip("# ").strip() for ln in md.splitlines() if ln.startswith("# ")), "Oppsummering")
    date_match = re.search(r"\*\*Dato:\*\*\s*(.+)", md)
    date = date_match.group(1).strip() if date_match else ""

    full_title = f"{title_line} — {date}" if date else title_line

    template = re.sub(
        r"<title>.*?</title>",
        f"<title>{full_title}</title>",
        template, count=1,
    )
    template = re.sub(
        r"<div>LOG650 — Gruppe 4\.5 — Møtereferat</div>",
        f"<div>LOG650 — Gruppe 4.5 — Oppsummering{(' — ' + date) if date else ''}</div>",
        template, count=1,
    )
    # Replace the embedded markdown
    template = re.sub(
        r'(<script type="text/x-markdown" id="md-source">\s*\n)(.*?)(\n</script>)',
        lambda m: m.group(1) + md + m.group(3),
        template, count=1, flags=re.DOTALL,
    )

    html_path = md_path.with_suffix(".html")
    pdf_path = md_path.with_suffix(".pdf")
    html_path.write_text(template, encoding="utf-8")
    print(f"Wrote {html_path.name} ({len(template):,} bytes)")

    # Convert to file:// URL for Chrome
    abs_html = html_path.resolve()
    url = "file:///" + str(abs_html).replace("\\", "/").replace(" ", "%20")

    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=8000",
         f"--print-to-pdf={pdf_path.resolve()}", url],
        capture_output=True, check=True,
    )
    print(f"Wrote {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python build_oppsummering.py <file.md> [<file.md> ...]")
    for arg in sys.argv[1:]:
        build(HERE / arg)
