"""Check the AI setup:  python -m pawpal_ai.check

Reports whether a Gemini key is configured, whether the model answers, and
whether Wikipedia is reachable. Never prints the key.
"""

from __future__ import annotations

import os
import sys

from .llm import DEFAULT_MODEL, LLMError, load_dotenv, make_client
from .retrieval import WebRetriever


def main() -> int:
    load_dotenv()
    ok = True

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("PAWPAL_MODEL") or DEFAULT_MODEL
    if not key:
        print("AI model : not configured -> the app runs in OFFLINE mode.")
        print("           Get a free key at https://aistudio.google.com/apikey and put")
        print("           GEMINI_API_KEY=your-key in a file named .env (see .env.example).")
    else:
        client = make_client()
        if client is None:
            print("AI model : key found but the client could not start (is google-genai installed?)")
            ok = False
        else:
            try:
                reply = client.generate("Reply with exactly the word: ready", "Are you ready?")
                print(f"AI model : OK - {client.name} replied {reply.strip()[:40]!r}")
            except LLMError as error:
                ok = False
                print(f"AI model : FAILED - {error}")
                print(f"           If the model {model!r} is not available on the free tier,")
                print("           set PAWPAL_MODEL in .env to a current Flash model from AI Studio.")

    try:
        results = WebRetriever().search("Siberian Husky", k=1)
        print(f"Web      : OK - Wikipedia returned {results[0].title!r}" if results
              else "Web      : reachable but returned nothing")
    except Exception as error:  # report, don't crash: the app works without the web
        print(f"Web      : unavailable ({type(error).__name__}: {error}) - the app will use local notes only")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
