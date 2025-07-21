#!/usr/bin/env python3
"""
Test runner script for the GPT UI project.
Provides convenient ways to run tests with different configurations.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return the exit code"""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)

def main():
    """Main test runner function"""
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
Test Runner for GPT UI Project

Usage:
    python run_tests.py [option]

Options:
    (no args)   Run working tests with coverage
    quick       Run working tests without coverage
    coverage    Run working tests and open coverage report
    all         Run ALL tests (including broken ones)
    install     Install test dependencies
    clean       Clean test artifacts
    
Examples:
    python run_tests.py            # Working tests with coverage
    python run_tests.py quick      # Quick working tests run
    python run_tests.py coverage   # Working tests + open coverage report
    python run_tests.py all        # All tests (some will fail)
        """)
        return 0
    
    option = sys.argv[1] if len(sys.argv) > 1 else 'default'
    
    if option == 'install':
        print("Installing test dependencies...")
        return run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-test.txt'])
    
    elif option == 'clean':
        print("Cleaning test artifacts...")
        import shutil
        artifacts = ['htmlcov', '.coverage', '.pytest_cache', '__pycache__']
        for artifact in artifacts:
            if os.path.exists(artifact):
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                    print(f"Removed directory: {artifact}")
                else:
                    os.remove(artifact)
                    print(f"Removed file: {artifact}")
        return 0
    
    elif option == 'quick':
        print("Running working tests (no coverage)...")
        return run_command([sys.executable, '-m', 'pytest', 'tests/test_simple.py', 'tests/test_fixed.py', '--no-cov', '-q'])
    
    elif option == 'unit':
        print("Running unit tests...")
        return run_command([sys.executable, '-m', 'pytest', '-m', 'unit'])
    
    elif option == 'integration':
        print("Running integration tests...")
        return run_command([sys.executable, '-m', 'pytest', '-m', 'integration'])
    
    elif option == 'coverage':
        print("Running working tests with coverage report...")
        exit_code = run_command([sys.executable, '-m', 'pytest', 'tests/test_simple.py', 'tests/test_fixed.py'])
        if exit_code == 0 and os.path.exists('htmlcov/index.html'):
            print("Opening coverage report...")
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath("htmlcov/index.html")}')
        return exit_code
    
    elif option == 'all':
        print("Running ALL tests (including ones that need fixes)...")
        return run_command([sys.executable, '-m', 'pytest'])
    
    elif option == 'default':
        print("Running working tests with coverage...")
        return run_command([sys.executable, '-m', 'pytest', 'tests/test_simple.py', 'tests/test_fixed.py'])
    
    else:
        print(f"Unknown option: {option}")
        print("Use 'python run_tests.py --help' for available options")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 