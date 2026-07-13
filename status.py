#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "rag_system"))

from src.cli.status import main

if __name__ == "__main__":
    main()
