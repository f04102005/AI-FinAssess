# Failure Analysis & Next Fixes

During the 20-question evaluation (`evaluate.py --questions questions.json`), the chatbot achieved 90.0% answer accuracy, 95.0% citation correctness, and 85.0% routing correctness. Below are the 3 specific failures observed and how I would fix them next.

### 1. Wrong Route on Policy Outlay Question (Q07)
- **What happened:** Asked *"What is the central outlay for the Prime Minister's package of 5 schemes for employment and skilling?"*. Because the prompt contained "outlay" and "schemes," the router sent it to `csv_lookup` instead of `doc_search`. Since this package is described in the PIB summary and budget speech rather than as a row in `budget_table.csv`, the lookup returned empty, retried once, and fell back to `"I don't have enough reliable information"`.
- **Fix:** Pass the exact list of `Item` names from `budget_table.csv` (or a compact summary of its 4 categories) into the `router` prompt, and update the retry logic so a failed `csv_lookup` automatically falls back to `doc_search` on the retry attempt.

### 2. Over-Aggressive Intent Refusal on Future Budget Year (Q17)
- **What happened:** Asked *"What is the Union Budget 2025-26 allocation for the Ministry of Space?"*. The `intent_check` node classified the mention of "2025-26" as `off_topic` and refused it in 1.38s instead of letting it pass to retrieval and returning the required `"I don't have enough reliable information"` fallback.
- **Fix:** Narrow the `intent_check` prompt so that `off_topic` only triggers for non-budget domains (sports, entertainment, coding, etc.). Any India Budget question—even for another fiscal year—should be marked `valid` so the retrieval + grader pipeline handles it with the standard insufficient-information response.

### 3. Unnecessary CSV Route & High Latency on Unanswerable Query (Q18)
- **What happened:** Asked *"What is the exact district-wise road construction budget for Coimbatore in 2024-25?"*. Triggered by "road" and "budget," the router sent it to `csv_lookup` instead of `doc_search`. While the system ultimately returned the correct `"I don't have enough reliable information"` response, running the retry loop pushed latency to 35.5s.
- **Fix:** Combine `intent_check` and `router` into a single structured LLM call to cut sequential latency across all turns, and add a lightweight lexical check in `csv_tool.py` so queries referencing entities absent from the CSV (like specific cities/districts) immediately short-circuit to `doc_search` without wasting a retry cycle.