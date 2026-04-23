#!/bin/bash
# Test the Apptainer container functionality
# Usage: ./test.sh

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Load configuration
if [[ ! -f "params.sh" ]]; then
    echo "Error: params.sh not found in current directory" >&2
    exit 1
fi
source params.sh

# Validate NAME variable
if [[ -z "${NAME:-}" ]]; then
    echo "Error: NAME variable not set in params.sh" >&2
    exit 1
fi

# Check if container file exists
if [[ ! -f "${NAME}.sif" ]]; then
    echo "Error: Container file ${NAME}.sif not found" >&2
    echo "Please run build.sh first to create the container" >&2
    exit 1
fi

# Check if apptainer command is available
if ! command -v apptainer &> /dev/null; then
    echo "Error: apptainer command not found. Please install Apptainer first." >&2
    exit 1
fi

echo "========================================="
echo "Testing container: ${NAME}.sif"
echo "========================================="
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run tests
run_test() {
    local test_name=$1
    local test_command=$2

    echo "Running test: ${test_name}"
    echo "Command: ${test_command}"

    if eval "${test_command}"; then
        echo "✓ PASSED: ${test_name}"
        ((TESTS_PASSED++))
    else
        echo "✗ FAILED: ${test_name}" >&2
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Test 1: Check if Python is available and from conda environment
run_test "Python availability" \
    "apptainer exec ${NAME}.sif which python"

# Test 2: Check Python version
run_test "Python version" \
    "apptainer exec ${NAME}.sif python --version"

# Test 3: Check if conda environment is activated
run_test "Conda environment activation" \
    "apptainer exec ${NAME}.sif bash -c 'echo \$CONDA_DEFAULT_ENV' | grep -q 'apptainer'"

# Test 4: Find jupyter-lab
run_test "JupyterLab availability" \
    "apptainer exec ${NAME}.sif which jupyter-lab"

# Test 5: Check JupyterLab version
run_test "JupyterLab version" \
    "apptainer exec ${NAME}.sif jupyter-lab --version"

# Test 6: Export conda environment (for verification)
echo "Exporting conda environment for verification..."
if apptainer exec "${NAME}.sif" conda env export --no-builds > environment_export.yml; then
    echo "✓ Environment exported to: environment_export.yml"
    echo ""
    ((TESTS_PASSED++))
else
    echo "✗ Failed to export environment" >&2
    echo ""
    ((TESTS_FAILED++))
fi

# Print summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Tests passed: ${TESTS_PASSED}"
echo "Tests failed: ${TESTS_FAILED}"
echo "========================================="

if [[ ${TESTS_FAILED} -eq 0 ]]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Some tests failed. Please review the output above." >&2
    exit 1
fi

