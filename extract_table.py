import re
import pdfplumber
import pandas as pd

ROW_PATTERN = re.compile(
    r"^(.+?)\s+([\d,\.]+)\s+([\d,\.]+)\s+([\d,\.]+)\s+([\d,\.]+)\s+(-?[\d\.]+)%$"
)

def clean_num(val):
    return float(val.replace(",", "").replace(".", "", val.count(".") - 1) if val.count(".") > 1 else val.replace(",", ""))

rows = []
current_category = None
current_basis = None

with pdfplumber.open("data/prs_analysis.pdf") as pdf:
    for page_num in [5, 6]:
        text = pdf.pages[page_num - 1].extract_text()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Table 4:"):
                current_category = "Expenditure Breakup"
                current_basis = "RE"
                continue
            elif line.startswith("Table 5:"):
                current_category = "Ministry"
                current_basis = "Actuals"
                continue
            elif line.startswith("Table 6:"):
                current_category = "Subsidy"
                current_basis = "Actuals"
                continue
            elif line.startswith("Table 7:"):
                current_category = "Scheme"
                current_basis = "RE"
                continue
            elif line.startswith("Sources:"):
                current_category = None
                continue

            if current_category:
                match = ROW_PATTERN.match(line)
                if match:
                    item, a22, b23, ra23, b24, pct = match.groups()
                    rows.append({
                        "Category": current_category,
                        "Item": item.strip(),
                        "Actuals_2022_23_Cr": clean_num(a22),
                        "Budgeted_2023_24_Cr": clean_num(b23),
                        "Revised_or_Actuals_2023_24_Cr": clean_num(ra23),
                        "Basis_2023_24": current_basis,
                        "Budgeted_2024_25_Cr": clean_num(b24),
                        "Pct_Change": float(pct),
                        "Source_Page": f"prs_analysis.pdf p.{page_num}"
                    })

df = pd.DataFrame(rows)
df.to_csv("budget_table.csv", index=False)
print(f"Saved budget_table.csv with {len(df)} rows.")
print(df.head(10).to_string())