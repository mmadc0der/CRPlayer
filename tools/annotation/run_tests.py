#!/usr/bin/env python3
"""
Test runner script for the annotation tool.

This script provides a convenient way to run different types of tests
with various configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
  """Run a command and handle errors."""
  print(f"\n{'='*60}")
  print(f"Running: {description}")
  print(f"Command: {' '.join(cmd)}")
  print(f"{'='*60}")

  try:
    result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
    print(f"✅ {description} completed successfully")
    return True
  except subprocess.CalledProcessError as e:
    print(f"❌ {description} failed with exit code {e.returncode}")
    return False
  except FileNotFoundError:
    print(f"❌ Command not found. Make sure pytest is installed: pip install -r requirements-test.txt")
    return False


def main():
  parser = argparse.ArgumentParser(description="Run tests for the annotation tool")
  parser.add_argument(
    "--type",
    choices=["all", "unit", "integration", "db", "api", "coverage"],
    default="all",
    help="Type of tests to run (default: all)",
  )
  parser.add_argument("--verbose", "-v", action="store_true", help="Run tests in verbose mode")
  parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel (specify number of workers)")
  parser.add_argument("--coverage-html", action="store_true", help="Generate HTML coverage report")
  parser.add_argument("--fast", action="store_true", help="Run tests without coverage for faster execution")
  parser.add_argument("--markers", "-m", help="Run tests with specific markers (e.g., 'not slow')")
  parser.add_argument("--pattern", "-k", help="Run tests matching pattern")

  args = parser.parse_args()

  # Base pytest command
  pytest_cmd = [sys.executable, "-m", "pytest"]

  # Add verbosity
  if args.verbose:
    pytest_cmd.append("-v")

  # Add parallel execution
  if args.parallel:
    pytest_cmd.extend(["-n", str(args.parallel)])

  # Add markers
  if args.markers:
    pytest_cmd.extend(["-m", args.markers])

  # Add pattern matching
  if args.pattern:
    pytest_cmd.extend(["-k", args.pattern])

  # Configure coverage
  if not args.fast:
    pytest_cmd.extend(["--cov=.", "--cov-branch"])
    if args.coverage_html:
      pytest_cmd.extend(["--cov-report=html:htmlcov"])

  # Determine test paths based on type
  test_paths = []
  if args.type == "all":
    test_paths = ["tests/"]
  elif args.type == "unit":
    test_paths = ["tests/unit/"]
  elif args.type == "integration":
    test_paths = ["tests/integration/"]
  elif args.type == "db":
    test_paths = ["tests/unit/db/"]
  elif args.type == "api":
    test_paths = ["tests/integration/api/"]
  elif args.type == "coverage":
    # Special coverage-only run
    pytest_cmd.extend([
      "--cov=.",
      "--cov-branch",
      "--cov-report=term-missing:skip-covered",
      "--cov-report=html:htmlcov",
      "--cov-report=xml",
      "tests/",
    ])
    test_paths = []

  # Add test paths
  pytest_cmd.extend(test_paths)

  # Run the tests
  success = run_command(pytest_cmd, f"Running {args.type} tests")

  if success:
    print(f"\n🎉 All tests passed!")
    if args.coverage_html:
      print(f"📊 Coverage report generated in htmlcov/index.html")
  else:
    print(f"\n💥 Some tests failed. Check the output above for details.")
    sys.exit(1)


if __name__ == "__main__":
  main()
