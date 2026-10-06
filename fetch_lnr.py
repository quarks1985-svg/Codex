from playwright.sync_api import sync_playwright
url = "https://chinalaser.spb.ru/ustanovka-lazernoj-rezki-metalla-lf3015lnr-pro/"
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--no-sandbox"])
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
    try:
        pg.goto(url, wait_until="domcontentloaded", timeout=45000)
        pg.wait_for_timeout(6000)  # pass JS-challenge
        html = pg.content()
        open("lnr_real.html","w",encoding="utf-8").write(html)
        print("LEN:", len(html))
        t = pg.title(); print("TITLE:", t)
    except Exception as e:
        print("ERR:", e)
    b.close()
