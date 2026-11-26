import os
from typing import Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DF = pd.read_csv("budget_table.csv")

VALID_COLUMNS = [
    "Actuals_2022_23_Cr",
    "Budgeted_2023_24_Cr",
    "Revised_or_Actuals_2023_24_Cr",
    "Budgeted_2024_25_Cr",
    "Pct_Change",
]


class CSVQueryPlan(BaseModel):
    operation: Literal["lookup", "sum", "compare", "percentage_change", "top_n"]
    items: list[str] = Field(default_factory=list)
    category: Optional[Literal["Expenditure Breakup", "Ministry", "Subsidy", "Scheme"]] = None
    columns: list[
        Literal[
            "Actuals_2022_23_Cr",
            "Budgeted_2023_24_Cr",
            "Revised_or_Actuals_2023_24_Cr",
            "Budgeted_2024_25_Cr",
            "Pct_Change",
        ]
    ] = Field(default_factory=lambda: ["Budgeted_2024_25_Cr"])
    top_n: int = 5


def match_rows(items: list[str], category: Optional[str]) -> pd.DataFrame:
    subset = DF.copy()
    if category:
        subset = subset[subset["Category"].str.lower() == category.lower()]
    if not items:
        return subset
    mask = pd.Series(False, index=subset.index)
    for item in items:
        mask = mask | subset["Item"].str.contains(item, case=False, na=False, regex=False)
    matched = subset[mask]
    if matched.empty and category:
        mask_all = pd.Series(False, index=DF.index)
        for item in items:
            mask_all = mask_all | DF["Item"].str.contains(item, case=False, na=False, regex=False)
        return DF[mask_all]
    return matched


def run_csv_query(question: str, llm) -> tuple[str, list[str]]:
    items_list = ", ".join(DF["Item"].unique().tolist())
    prompt = (
        "Convert the user question into a structured CSV query plan.\n"
        f"Available Categories: Expenditure Breakup, Ministry, Subsidy, Scheme\n"
        f"Available Columns: {', '.join(VALID_COLUMNS)}\n"
        f"Available Items: {items_list}\n"
        "Pick the closest matching item names from Available Items. "
        "Use percentage_change when asked for percent increase/decrease between two columns. "
        f"Question: {question}"
    )

    planner = llm.with_structured_output(CSVQueryPlan)
    plan: CSVQueryPlan = planner.invoke(prompt)

    matched = match_rows(plan.items, plan.category)
    if matched.empty:
        return "", []

    cols = [c for c in plan.columns if c in VALID_COLUMNS] or ["Budgeted_2024_25_Cr"]
    lines = []
    sources = []

    if plan.operation == "lookup":
        for idx, row in matched.iterrows():
            vals = ", ".join(f"{c}: {row[c]}" for c in cols)
            lines.append(f"{row['Item']} ({row['Category']}, 2023-24 basis: {row['Basis_2023_24']}): {vals}")
            sources.append(f"budget_table.csv (Row {idx + 2}, Cols: {', '.join(cols)})")

    elif plan.operation == "sum":
        col = cols[0]
        total = matched[col].sum()
        item_names = ", ".join(matched["Item"].tolist())
        lines.append(f"Sum of {col} for [{item_names}] = {total:,.2f} Rs crore")
        row_nums = ", ".join(str(idx + 2) for idx in matched.index)
        sources.append(f"budget_table.csv (Rows {row_nums}, Col: {col})")

    elif plan.operation == "compare":
        col = cols[0]
        sorted_df = matched.sort_values(by=col, ascending=False)
        for idx, row in sorted_df.iterrows():
            lines.append(f"{row['Item']}: {row[col]:,.2f} Rs crore ({col})")
            sources.append(f"budget_table.csv (Row {idx + 2}, Col: {col})")
        if len(sorted_df) == 2:
            diff = sorted_df.iloc[0][col] - sorted_df.iloc[1][col]
            lines.append(f"Difference ({sorted_df.iloc[0]['Item']} - {sorted_df.iloc[1]['Item']}): {diff:,.2f} Rs crore")

    elif plan.operation == "percentage_change":
        base_col = cols[0] if len(cols) >= 2 else "Revised_or_Actuals_2023_24_Cr"
        target_col = cols[1] if len(cols) >= 2 else "Budgeted_2024_25_Cr"
        for idx, row in matched.iterrows():
            base_val = row[base_col]
            target_val = row[target_col]
            if base_val != 0:
                pct = ((target_val - base_val) / base_val) * 100
                lines.append(
                    f"{row['Item']}: {base_col}={base_val:,.2f}, {target_col}={target_val:,.2f}, "
                    f"Change={pct:.2f}% (Table Pct_Change: {row['Pct_Change']}%)"
                )
            else:
                lines.append(f"{row['Item']}: {base_col}=0, {target_col}={target_val:,.2f}")
            sources.append(f"budget_table.csv (Row {idx + 2}, Cols: {base_col}, {target_col})")

    elif plan.operation == "top_n":
        col = cols[0]
        filtered = matched[~matched["Item"].str.contains("Total|Other Ministries", case=False, na=False)]
        top_df = filtered.sort_values(by=col, ascending=False).head(plan.top_n)
        for idx, row in top_df.iterrows():
            lines.append(f"{row['Item']} ({row['Category']}): {row[col]:,.2f} Rs crore ({col})")
            sources.append(f"budget_table.csv (Row {idx + 2}, Col: {col})")

    return "\n".join(lines), sources


if __name__ == "__main__":
    test_llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        temperature=0,
    )
    for q in [
        "What is the 2024-25 budgeted allocation for Defence?",
        "What is the combined 2024-25 allocation for Railways and Road Transport and Highways?",
        "Top 3 schemes by 2024-25 budgeted allocation",
    ]:
        ctx, src = run_csv_query(q, test_llm)
        print(f"\nQ: {q}\n{ctx}\nSources: {src}")