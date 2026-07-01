import json
import logging
import os
import time
import urllib.request
import urllib.error

import azure.functions as func


ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "").rstrip("/")
KEY = os.environ.get("LANGUAGE_KEY", "")

API_VERSION = "2023-04-01"
TIMEOUT_SECONDS = 20


# =========================
# MAIN FUNCTION
# =========================
def main(req: func.HttpRequest) -> func.HttpResponse:

    if not ENDPOINT or not KEY:
        return _json_response(
            {
                "error": "Missing LANGUAGE_ENDPOINT or LANGUAGE_KEY in environment variables."
            },
            500,
        )

    try:
        body = req.get_json()
    except Exception:
        body = {}

    # ✅ SAFE INPUT HANDLING
    text = body.get("text")

    if not isinstance(text, str) or not text.strip():
        return _json_response(
            {"error": "Invalid or missing 'text' field."},
            400,
        )

    text = text.strip()

    try:
        # =========================
        # CALL AZURE AI LANGUAGE
        # =========================
        result = _call_language(text)

        # =========================
        # SUMMARY (optional but safe)
        # =========================
        summary = _call_abstractive_summary(text)

        response = {
            "language": result.get("language", {}),
            "sentiment": result.get("sentiment"),
            "confidenceScores": result.get("confidenceScores", {}),
            "keyPhrases": result.get("keyPhrases", []),
            "entities": result.get("entities", []),
            "piiEntities": result.get("piiEntities", []),
            "redactedText": result.get("redactedText", ""),
            "summary": summary,
        }

        return _json_response(response, 200)

    except Exception:
        logging.exception("Azure AI Language call failed")
        return _json_response(
            {"error": "Azure AI Language request failed."},
            502,
        )


# =========================
# SINGLE BATCH CALL (FIXED)
# =========================
def _call_language(text: str) -> dict:

    url = f"{ENDPOINT}/language/:analyze-text?api-version={API_VERSION}"

    payload = {
        "displayName": "multi-analysis",
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "text": text,
                    "language": "en"
                }
            ]
        },
        "tasks": [
            {"kind": "SentimentAnalysis"},
            {"kind": "KeyPhraseExtraction"},
            {"kind": "EntityRecognition"},
            {"kind": "PiiEntityRecognition"},
            {"kind": "LanguageDetection"}
        ]
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = json.loads(response.read().decode())

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(detail)

    # =========================
    # PARSE RESPONSE SAFELY
    # =========================
    results = {}

    try:
        task_results = raw["results"]["tasks"]["items"]

        for task in task_results:

            kind = task.get("kind")

            docs = task.get("results", {}).get("documents", [])
            if not docs:
                continue

            doc = docs[0]

            if kind == "SentimentAnalysis":
                results["sentiment"] = doc.get("sentiment")
                results["confidenceScores"] = doc.get("confidenceScores", {})

            elif kind == "KeyPhraseExtraction":
                results["keyPhrases"] = doc.get("keyPhrases", [])

            elif kind == "EntityRecognition":
                results["entities"] = [
                    {
                        "text": e.get("text"),
                        "category": e.get("category"),
                    }
                    for e in doc.get("entities", [])
                ]

            elif kind == "PiiEntityRecognition":
                results["piiEntities"] = [
                    {
                        "text": e.get("text"),
                        "category": e.get("category"),
                    }
                    for e in doc.get("entities", [])
                ]
                results["redactedText"] = doc.get("redactedText", "")

            elif kind == "LanguageDetection":
                results["language"] = doc.get("detectedLanguage", {})

    except Exception as e:
        logging.exception("Parsing error: %s", str(e))

    return results


# =========================
# SUMMARY (FIXED POLLING)
# =========================
def _call_abstractive_summary(text: str):

    url = f"{ENDPOINT}/language/analyze-text/jobs?api-version={API_VERSION}"

    payload = {
        "displayName": "summary",
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": text
                }
            ]
        },
        "tasks": [
            {
                "kind": "AbstractiveSummarization",
                "parameters": {
                    "sentenceCount": 3
                }
            }
        ]
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": KEY,
        },
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        operation_url = response.headers.get("operation-location")

    if not operation_url:
        return []

    # Polling
    while True:

        poll_request = urllib.request.Request(
            operation_url,
            headers={"Ocp-Apim-Subscription-Key": KEY},
        )

        with urllib.request.urlopen(poll_request) as response:
            result = json.loads(response.read().decode())

        status = result.get("status", "").lower()

        if status == "succeeded":

            docs = result.get("tasks", {}).get("items", [])
            if not docs:
                return []

            summaries = docs[0].get("results", {}).get("documents", [])

            if summaries:
                return [
                    s.get("text")
                    for s in summaries[0].get("summaries", [])
                ]

            return []

        if status == "failed":
            raise RuntimeError("Summarization failed.")

        time.sleep(2)


# =========================
# RESPONSE HELPER
# =========================
def _json_response(payload: dict, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )