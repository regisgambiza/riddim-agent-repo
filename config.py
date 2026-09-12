"""
Central configuration for the riddim archive investigation agent.

Edit the values below (or override with environment variables of the
same name) to match your machine. Nothing in this file performs any
file-write or database-write operations by itself.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# App root = folder this config.py lives in.
APP_ROOT = Path(__file__).resolve().parent

# The SQLite database of known riddims. Read-only access only.
DB_PATH = Path(os.environ.get("RIDDIM_DB_PATH", APP_ROOT / "riddims_multi_year_new.db"))

# The folder containing source sub-folders to be investigated.
# Each immediate sub-folder of SOURCE_ROOT is treated as one "case".
SOURCE_ROOT = Path(os.environ.get("RIDDIM_SOURCE_ROOT", r"E:\Music" if Path(r"E:\Music").exists() else APP_ROOT / "Folder1"))

# Where match_proposals.json (the agent's only output) gets written.
OUTPUT_PATH = Path(os.environ.get("RIDDIM_OUTPUT_PATH", APP_ROOT / "match_proposals.json"))

# --------------------------------------------------------------------------
# Spotify API credentials (for enrichment + agent Spotify tool)
# --------------------------------------------------------------------------
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "072209652b0243f3b6f357577e103adf")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "2a868134eeab4a9db5a43e4fb3ce610c")
SPOTIFY_CACHE_PATH = Path(os.environ.get("SPOTIFY_CACHE_PATH", APP_ROOT / ".spotify_cache.db"))

# --------------------------------------------------------------------------
# llama.cpp server
# --------------------------------------------------------------------------

# Path to the llama-server executable. On Windows this is typically
# llama-server.exe from a llama.cpp release build, or a self-built binary.
# If left as "llama-server", it must be reachable on PATH.
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")

# Path to the GGUF model file that llama-server will load.
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    r"C:\Users\regis\.lmstudio\models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf",
)

# Host/port the agent will start llama-server on and talk to.
LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8090"))

# Context window to request from llama-server (-c). Tool-calling transcripts
# can get long across many tool calls per folder (each get_candidates /
# compare_track_evidence result adds a chunk of JSON to history), so this is
# generous. If you see "Failed to parse tool call arguments as JSON" errors
# from llama-server, the context filled up mid-generation and the model's
# output got truncated -- raise this further if your hardware allows it.
# The tool schema and investigation prompt alone can exceed 20K tokens, so
# retain a 32K window. Keeping one parallel slot limits KV-cache VRAM use.
LLAMA_CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", "32768"))

# Max tokens the model may generate in a single reply. Tool-call replies are
# compact; a 2K cap avoids spending minutes generating an oversized response.
LLAMA_MAX_TOKENS = int(os.environ.get("LLAMA_MAX_TOKENS", "2048"))

# GPU offload layers (-ngl). -1 lets llama.cpp decide / offload all it can.
# Set to 0 to force CPU-only.
LLAMA_N_GPU_LAYERS = int(os.environ.get("LLAMA_N_GPU_LAYERS", "-1"))

# One active request avoids reserving four independent KV caches, which would
# crowd model layers out of the GPU on an 8 GB card.
LLAMA_PARALLEL_SLOTS = int(os.environ.get("LLAMA_PARALLEL_SLOTS", "1"))

# Extra raw CLI args appended verbatim to the llama-server invocation.
# --jinja is required for proper tool-calling chat-template handling.
# --reasoning off disables internal CoT (<think>) token generation for models
# like Qwen3.6 that have reasoning templates, keeping tool calls fast and direct.
LLAMA_EXTRA_ARGS = os.environ.get("LLAMA_EXTRA_ARGS", "--jinja --reasoning off").split()

# How long (seconds) to wait for llama-server's /health endpoint before
# giving up on startup.
LLAMA_STARTUP_TIMEOUT_S = int(os.environ.get("LLAMA_STARTUP_TIMEOUT_S", "180"))

# --------------------------------------------------------------------------
# Agent behavior
# --------------------------------------------------------------------------

# Sampling temperature for the investigation agent. Kept low because this
# is a factual matching task, not creative generation.
LLAMA_TEMPERATURE = float(os.environ.get("LLAMA_TEMPERATURE", "0.1"))

# Hard safety cap on tool-call round-trips per folder. This is NOT meant to
# rush the agent -- it is a runaway-loop failsafe only. If ever hit, the
# folder is force-logged as needs_review with a note, never as a guessed
# match, so a bug never silently produces a false "matched" result.
MAX_TOOL_ROUNDTRIPS_PER_FOLDER = int(os.environ.get("MAX_TOOL_ROUNDTRIPS_PER_FOLDER", "40"))

# Default number of candidates get_candidates() returns unless overridden.
DEFAULT_TOP_N = int(os.environ.get("RIDDIM_DEFAULT_TOP_N", "8"))

# If a chat request to llama-server fails with a transient/server error
# (e.g. the model's JSON got truncated by a full context window), retry
# this many times with backoff before giving up on the *current round*.
# This does NOT retry a whole folder -- it just gives one shaky generation
# another chance. Exhausting retries counts as a normal failed round and
# feeds an error back to the model like any other tool error.
MAX_CHAT_RETRIES = int(os.environ.get("MAX_CHAT_RETRIES", "3"))
CHAT_RETRY_BACKOFF_S = float(os.environ.get("CHAT_RETRY_BACKOFF_S", "2.0"))

# Maximum consecutive LLM chat failures (e.g. server 500 / parse errors)
# before the folder is force-logged as needs_review, preventing runaway loops
# that waste minutes on a stuck prompt.
MAX_CONSECUTIVE_LLM_ERRORS = int(os.environ.get("MAX_CONSECUTIVE_LLM_ERRORS", "2"))

# main.py writes match_proposals.json after EVERY folder (not just at the
# end), so a crash partway through a large batch (e.g. 1800+ folders) never
# loses completed work. On the next run, folders already present in the
# output file are skipped automatically unless --no-resume is passed.
CHECKPOINT_EVERY_FOLDER = True

# Audio file extensions used by get_folder_contents() to split audio vs other.
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".wma",
    ".aac", ".alac", ".opus", ".mid", ".midi",
}
