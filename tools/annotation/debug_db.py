#!/usr/bin/env python3
"""
Debug script for database connection issues on Linux.
Run this to diagnose database problems.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from db.connection import get_db_path, get_connection
from db.schema import init_db

def debug_database():
    print("=== Database Debug Information ===")

    # Check database path
    db_path = get_db_path()
    print(f"Database path: {db_path}")
    print(f"Path exists: {db_path.exists()}")

    if db_path.exists():
        stat = db_path.stat()
        print(f"File size: {stat.st_size} bytes")
        print(f"File permissions: {oct(stat.st_mode)}")
        print(f"File owner: {stat.st_uid}:{stat.st_gid}")

    # Check parent directory
    parent_dir = db_path.parent
    print(f"Parent directory: {parent_dir}")
    print(f"Parent exists: {parent_dir.exists()}")

    if parent_dir.exists():
        parent_stat = parent_dir.stat()
        print(f"Parent permissions: {oct(parent_stat.st_mode)}")
        print(f"Parent owner: {parent_stat.st_uid}:{parent_stat.st_gid}")

    # Try to connect to database
    print("\n=== Testing Database Connection ===")
    try:
        conn = get_connection()
        print("✅ Database connection successful")

        # Check if tables exist
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        print(f"Tables in database: {[row[0] for row in tables]}")

        # Test basic operations
        print("\n=== Testing Basic Operations ===")
        try:
            # Test project creation
            from db.projects import create_project
            test_name = "debug_test_project"
            test_desc = "Debug test project"

            pid = create_project(conn, test_name, test_desc)
            print(f"✅ Created test project with ID: {pid}")

            # Clean up
            conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
            conn.commit()
            print("✅ Cleaned up test project")

        except Exception as e:
            print(f"❌ Basic operations failed: {e}")

        conn.close()
        print("✅ Database connection closed successfully")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_database()
