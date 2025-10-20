"""
Multi-Asset Trading Bot - Main Entry Point
===========================================
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main bot
from src.main_bot import main

if __name__ == '__main__':
    main()

