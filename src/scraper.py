# Librerías estándar y de terceros necesarias para scraping, análisis y manipulación
import time
import json
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Selenium y configuración del driver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Inicializamos el analizador de sentimientos de VADER
analyzer = SentimentIntensityAnalyzer()


# Inicialización del WebDriver
def init_driver():
    options = Options()

    # NOTA: Se comenta headless para poder resolver CAPTCHAS manualmente si aparecen
    # options.add_argument("--headless=new")

    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Cambiamos el user-agent para simular un navegador real
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "KHTML, like Gecko Chrome/123.0.0.0 Safari/537.36"
    )

    # Opciones para reducir detección de Selenium
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Inicializamos el driver de Chrome con configuración
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Ocultamos la propiedad "webdriver" en JavaScript
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver

# Cargar cookies desde archivo para evitar login manual
def load_cookies(driver, cookie_file="glassdoor_cookies.json"):
    print("Cargando cookies guardadas...")
    driver.get("https://www.glassdoor.com/")
    time.sleep(3)

    with open(cookie_file, "r") as f:
        cookies = json.load(f)
        for cookie in cookies:
            if "sameSite" in cookie:
                cookie["sameSite"] = 'Strict'  # Ajuste de compatibilidad para Selenium
            driver.add_cookie(cookie)

    # Refrescamos la página con las cookies aplicadas
    driver.refresh()
    time.sleep(4)


# Clasificar sentimiento usando VADER
def classify_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score > 0.02:
        return "Pro"
    elif score < -0.02:
        return "Con"
    else:
        return "Neutral"

# Scraping de reseñas de una empresa desde Glassdoor
def scrape_company_reviews(company_slug, max_reviews=100):
    print(f"Scrapeando reseñas para {company_slug}...")
    driver = init_driver()
    load_cookies(driver)

    reviews = []
    page = 1

    while len(reviews) < max_reviews:
        # Construir URL de la página correspondiente
        url = f"https://www.glassdoor.com/Reviews/{company_slug.replace('.htm', f'_P{page}.htm')}"
        driver.get(url)
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Buscamos los textos de Pros y Cons en la página
        pros = soup.find_all("span", attrs={"data-test": "review-text-PROS"})
        cons = soup.find_all("span", attrs={"data-test": "review-text-CONS"})

        if not pros and not cons:
            print(f"Página {page} no contiene reseñas visibles. Fin del scraping.")
            break

        # Emparejamos Pros y Cons por índice
        total = min(len(pros), len(cons))
        for i in range(total):
            if len(reviews) >= max_reviews:
                break
            try:
                pros_text = pros[i].get_text(strip=True)
                lang_pros = detect(pros_text)
                reviews.append({
                    "company": company_slug.split("-Reviews")[0],
                    "review": pros_text,
                    "language": lang_pros,
                    "type": "Pro"
                })
            except:
                pass

            if len(reviews) >= max_reviews:
                break
            try:
                cons_text = cons[i].get_text(strip=True)
                lang_cons = detect(cons_text)
                reviews.append({
                    "company": company_slug.split("-Reviews")[0],
                    "review": cons_text,
                    "language": lang_cons,
                    "type": "Con"
                })
            except:
                pass

        print(f"Página {page}: {total * 2} reseñas procesadas (Total: {len(reviews)})")
        page += 1

    driver.quit()
    print(f"Total reseñas extraídas para {company_slug}: {len(reviews)}")
    return reviews


# Scraping de múltiples empresas y guardado en CSV
def scrape_multiple(company_slugs, output_csv="data/glassdoor_reviews.csv"):
    all_reviews = []
    for slug in company_slugs:
        all_reviews.extend(scrape_company_reviews(slug))

    df = pd.DataFrame(all_reviews)
    df.to_csv(output_csv, index=False)
    print(f"Reseñas guardadas en {output_csv}")
