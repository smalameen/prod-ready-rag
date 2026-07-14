#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none python3 -m streamlit run rag_system/ui/app.py --server.port 3001 --server.headless true
