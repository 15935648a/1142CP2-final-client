#!/usr/bin/env python3
"""Large-network training entry point. Same as main.py but uses ConfigLarge."""
import sys
import main as _main
from connect6s.config_large import ConfigLarge

# Patch Config reference before main() reads it
_main.Config = ConfigLarge

if __name__ == "__main__":
    _main.main()
