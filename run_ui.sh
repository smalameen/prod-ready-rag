#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m streamlit run rag_system/ui/app.py
