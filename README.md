# Legal-Rag

## Query classification and retrieval

Queries are analyzed with deterministic intent/difficulty signals and then
classified by the configured general-purpose Groq model. If the model call
fails or returns an invalid classification, the rule-based result is used.
EASY and MEDIUM queries retrieve once; HARD queries can retrieve up to three
times. Before another HARD-query round, the model checks whether the collected
passages are sufficient and, when needed, proposes a focused follow-up search.
Round limits are enforced by the application, and evidence-assessment failures
stop further retrieval while preserving any passages already found.
