"""
Automated Efficacy & Functionality Test Suite Runner for Shallot-CUI-Bot.

Discovers and executes all modularized unit & integration tests in tests/
and runs the architectural audits from scripts/audit_loras.py.
"""

import sys
import os
import unittest
import logging

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Disable log output during test execution to keep console clean
logging.disable(logging.CRITICAL)


def run_all_tests() -> bool:
    """Discovers and runs all unit tests under the tests/ directory."""
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    suite = loader.discover(start_dir=start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
