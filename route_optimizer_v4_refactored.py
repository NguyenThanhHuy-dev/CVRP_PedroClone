#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Wrapper for route_optimizer_v4_refactored
==========================================
This file provides a wrapper to allow route_optimizer_v4_refactored folder
to be used with benchmark_runner_v2.py as if it were a single module file.

It simply imports and re-exports the solve_with_clarke_wright_and_optimize function
from the refactored folder's main.py
"""

import sys
import os

# Add the refactored folder to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
refactored_dir = os.path.join(current_dir, 'route_optimizer_v4_refactored')
sys.path.insert(0, refactored_dir)

# Import the main solve function from refactored main.py
from main import solve_with_clarke_wright_and_optimize

# Re-export for use by benchmark_runner_v2.py
__all__ = ['solve_with_clarke_wright_and_optimize']
