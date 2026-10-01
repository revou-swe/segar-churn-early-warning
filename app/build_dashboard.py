"""Inject the latest scoring output into the dashboard template.

    python app/build_dashboard.py      ->  app/dashboard.html (self-contained, open in any browser)

The HTML has no server and no external data calls: the scored outlets are embedded,
so the file can be emailed, dropped in Google Drive, or published as a Claude artifact.
"""
from pathlib import Path

APP = Path(__file__).resolve().parent


def build(standalone: bool = True) -> str:
    tpl = (APP / "dashboard_template.html").read_text(encoding="utf-8")
    data = (APP / "dashboard_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
    body = tpl.replace("/*__DATA__*/null", data)
    if not standalone:
        return body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + body + "\n</html>\n")


if __name__ == "__main__":
    out = APP / "dashboard.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out.relative_to(APP.parent)} ({out.stat().st_size/1024:,.0f} KB)")
