# Solo se corre una vez para guardar las cookies de Glassdoor

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

# Inicia navegador para login manual
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://www.glassdoor.com.mx/index.htm?countryRedirect=true")

print("Por favor inicia sesión manualmente...")
time.sleep(60)  # Da tiempo para iniciar sesión manual

# Guarda cookies en archivo JSON
cookies = driver.get_cookies()
with open("glassdoor_cookies.json", "w") as f:
    json.dump(cookies, f)

print("Cookies guardadas en glassdoor_cookies.json")
driver.quit()
