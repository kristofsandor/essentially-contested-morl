#!/bin/bash
# Check if all required dependencies are installed
# Usage: ./check-dependencies.sh

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

echo "========================================="
echo "Checking System Dependencies"
echo "========================================="
echo ""

# Track overall status
ALL_OK=true

# Function to check if a command exists
check_command() {
    local cmd=$1
    local version_flag=${2:-"--version"}
    local required=${3:-true}

    echo -n "Checking for ${cmd}... "

    if command -v "${cmd}" &> /dev/null; then
        echo "✓ found"

        # Try to get version info
        if [[ "${version_flag}" != "SKIP" ]]; then
            local version_output
            version_output=$(${cmd} ${version_flag} 2>&1 | head -n 1 || echo "version unknown")
            echo "  Version: ${version_output}"
        fi
        return 0
    else
        if [[ "${required}" == true ]]; then
            echo "✗ NOT FOUND (required)"
            ALL_OK=false
        else
            echo "⚠ not found (optional)"
        fi
        return 1
    fi
}

# Check required commands
echo "Required Dependencies:"
echo "---------------------"
check_command "apptainer" "--version"
check_command "rsync" "--version"
check_command "git" "--version"
check_command "bash" "--version"

echo ""
echo "Optional Dependencies:"
echo "---------------------"
check_command "make" "--version" false
check_command "yamllint" "--version" false

echo ""
echo "Checking System Resources:"
echo "-------------------------"

# Check disk space in current directory
AVAILABLE_SPACE=$(df -BG . | tail -n 1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_SPACE}GB"

if [[ ${AVAILABLE_SPACE} -lt 10 ]]; then
    echo "  ⚠ Warning: Less than 10GB available. Container builds may require significant space."
fi

# Check if we're in a git repository
echo ""
echo "Git Repository Status:"
echo "---------------------"
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✓ Git repository detected"

    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        echo "  ⚠ Warning: You have uncommitted changes"
    else
        echo "  ✓ Working directory is clean"
    fi
else
    echo "⚠ Not a git repository"
    echo "  Version tracking will be limited"
fi

# Final summary
echo ""
echo "========================================="
if [[ "${ALL_OK}" == true ]]; then
    echo "✓ All required dependencies are installed!"
    echo "========================================="
    exit 0
else
    echo "✗ Some required dependencies are missing!"
    echo "Please install the missing dependencies before proceeding."
    echo "========================================="
    exit 1
fi
