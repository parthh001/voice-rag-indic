from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subset"
RESULTS_DIR = ROOT_DIR / "results"

STT_PROVIDER = os.environ.get("STT_PROVIDER", "mock")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "mock")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
DATASET_LANG = os.environ.get("DATASET_LANG", "hin")
CHROMA_PATH = os.environ.get("CHROMA_PATH", ".chroma")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
