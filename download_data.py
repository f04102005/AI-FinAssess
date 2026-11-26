import os
import requests

os.makedirs("data", exist_ok=True)

FILES = {
    "budget_speech.docx": "https://indiabudget.gov.in/budget2024-25/doc/budget_speech.docx",
    "bag1.pdf": "https://indiabudget.gov.in/budget2024-25/doc/Budget_at_Glance/bag1.pdf",
    "hkeybud2024.pdf": "https://indiabudget.gov.in/budget2024-25/doc/hkeybud2024.pdf",
    "pib_summary.pdf": "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2024/jul/doc2024723352501.pdf",
    "prs_analysis.pdf": "https://prsindia.org/files/budget/budget_parliament/2024/Union_Budget_Analysis_2024-25.pdf",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

for filename, url in FILES.items():
    path = os.path.join("data", filename)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {filename}")