"""PawPal Assistant: AI features (RAG + agent) on top of the PawPal+ scheduler."""

from .advisor import Answer, PawPalAdvisor
from .llm import make_client
from .logging_setup import setup_logging
from .retrieval import Retriever, WebRetriever

__all__ = ["Answer", "PawPalAdvisor", "Retriever", "WebRetriever", "make_client", "setup_logging"]
