"""Entry point for preprocessing script (numbered for pipeline ordering)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_data import main

if __name__ == "__main__":
    main()
