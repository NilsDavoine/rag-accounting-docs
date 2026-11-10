"""Screenshots for the README: one answered question with its source, one refusal.

Needs the web page running (`make web`) and Playwright (`pip install playwright`); it drives an
installed Chrome, so no browser download is needed.

  python docs/screenshot.py [--chrome /usr/bin/google-chrome] [--url http://localhost:8000]"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
SHOTS = [("answer.png", "ombrelune", "Quel taux horaire avons-nous convenu avec Cabinet Vauclerc Conseil ?"),
         ("refusal.png", "quillfern", "What is the total of invoice VCC-2025-711?")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome)
        page = browser.new_page(viewport={"width": 860, "height": 900}, device_scale_factor=2, color_scheme="light")
        # one question first, so the screenshots show warm timings rather than model loading
        page.goto(args.url)
        page.request.post(args.url + "/ask", data={"company": "halvardt", "question": "Wann beginnt der Vertrag?"})
        for name, company, question in SHOTS:
            page.goto(args.url)
            page.select_option("#c", company)
            page.fill("#q", question)
            page.click("button")
            page.wait_for_function("(t => t && !t.startsWith('…'))(document.querySelector('#out').innerText.trim())",
                                   timeout=120_000)
            page.locator("main").screenshot(path=str(OUT / name))
            print("wrote", OUT / name)
        browser.close()


if __name__ == "__main__":
    main()
