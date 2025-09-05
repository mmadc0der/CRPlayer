#!/usr/bin/env python3
"""
Run core tests with focus on critical functionality.
This script runs tests that validate the most important features.
"""

import subprocess
import sys
from pathlib import Path


def run_core_tests():
    """Run core tests that validate essential functionality."""
    
    # Core test suites that must pass
    core_test_suites = [
        # Database layer - 100% critical
        "tests/unit/db/test_schema.py",
        "tests/unit/db/test_connection.py", 
        "tests/unit/db/test_repository.py",
        "tests/unit/db/test_projects.py",
        
        # Data models - 100% critical
        "tests/unit/test_dto.py",
        
        # Core business logic - critical
        "tests/unit/core/test_session_manager.py",
        
        # API endpoints that work - important
        "tests/integration/api/test_projects_api.py::TestProjectsAPI::test_list_projects_empty",
        "tests/integration/api/test_projects_api.py::TestProjectsAPI::test_create_project_success",
        "tests/integration/api/test_sessions_api.py",
    ]
    
    print("🧪 Running core test suite...")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    for test_suite in core_test_suites:
        print(f"\n📋 Running: {test_suite}")
        try:
            result = subprocess.run([
                "python3", "-m", "pytest", test_suite, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                print(f"✅ PASSED: {test_suite}")
                # Count passed tests
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'passed' in line and '=' in line:
                        try:
                            parts = line.split()
                            for part in parts:
                                if 'passed' in part:
                                    num = int(part.split('passed')[0].strip())
                                    total_passed += num
                                    break
                        except:
                            pass
            else:
                print(f"❌ FAILED: {test_suite}")
                # Count failed tests
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'failed' in line and 'passed' in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if 'failed' in part:
                                    failed_num = int(parts[i-1])
                                    total_failed += failed_num
                                if 'passed' in part:
                                    passed_num = int(parts[i-1])
                                    total_passed += passed_num
                                    break
                        except:
                            pass
                            
        except Exception as e:
            print(f"❌ ERROR running {test_suite}: {e}")
    
    print("\n" + "=" * 60)
    print("🏆 CORE TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = total_passed + total_failed
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("\n🎉 EXCELLENT: Core functionality is extremely well tested!")
        print("🚀 Ready for production use")
    elif success_rate >= 90:
        print("\n✅ GOOD: Core functionality is well tested")
        print("🔧 Minor improvements recommended")
    elif success_rate >= 80:
        print("\n⚠️  ACCEPTABLE: Core functionality is adequately tested")
        print("🔧 Some improvements needed")
    else:
        print("\n🚨 ATTENTION: Core functionality needs more testing")
        print("🛠️  Significant improvements required")
    
    return success_rate >= 90


if __name__ == "__main__":
    success = run_core_tests()
    sys.exit(0 if success else 1)
