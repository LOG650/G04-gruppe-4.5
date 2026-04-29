"""Rebuild the HTML preview from the current markdown source.

Replaces the embedded markdown payload between the
<script type="text/x-markdown" id="md-source"> ... </script>
tags in the existing HTML file, and updates title/banner text
to reflect the new filename.
"""
from pathlib import Path
import re

HERE = Path(__file__).parent
MD = HERE / "Forskningsoppgave_Gruppe_4.5.md"
HTML = HERE / "Forskningsoppgave_Gruppe_4.5.html"

md_text = MD.read_text(encoding="utf-8")
html_text = HTML.read_text(encoding="utf-8")

# Update title and banner
html_text = html_text.replace(
    "<title>Bacheloroppgave – Skoringen Råholt (forhåndsvisning)</title>",
    "<title>Forskningsoppgave Gruppe 4.5 – Skoringen Råholt (forhåndsvisning)</title>",
)
html_text = html_text.replace(
    "<div>LOG650 — Bacheloroppgave (forhåndsvisning)</div>",
    "<div>LOG650 — Forskningsoppgave Gruppe 4.5 (forhåndsvisning)</div>",
)

# Replace markdown payload
pattern = re.compile(
    r'(<script type="text/x-markdown" id="md-source">\s*\n)(.*?)(\n</script>)',
    re.DOTALL,
)
new_html, n = pattern.subn(lambda m: m.group(1) + md_text + m.group(3), html_text)
if n != 1:
    raise SystemExit(f"Expected exactly one md-source block, found {n}")

HTML.write_text(new_html, encoding="utf-8")
print(f"Wrote {HTML.name}: {len(new_html):,} bytes (md payload {len(md_text):,} bytes)")
