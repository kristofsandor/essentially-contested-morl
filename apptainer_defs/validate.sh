#!/bin/bash
# Validate configuration and files before building
# Usage: ./validate.sh

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

echo "========================================="
echo "Validating Build Configuration"
echo "========================================="
echo ""

# Track validation status
VALIDATION_OK=true

# Function to report validation results
report_check() {
    local check_name=$1
    local status=$2
    local message=${3:-""}

    if [[ "${status}" == "pass" ]]; then
        echo "✓ ${check_name}"
        if [[ -n "${message}" ]]; then
            echo "  ${message}"
        fi
    elif [[ "${status}" == "warn" ]]; then
        echo "⚠ ${check_name}"
        if [[ -n "${message}" ]]; then
            echo "  ${message}"
        fi
    else
        echo "✗ ${check_name}"
        if [[ -n "${message}" ]]; then
            echo "  ${message}"
        fi
        VALIDATION_OK=false
    fi
}

# Check 1: Required files exist
echo "Checking Required Files:"
echo "------------------------"

if [[ -f "params.sh" ]]; then
    report_check "params.sh exists" "pass"
else
    report_check "params.sh exists" "fail" "File not found"
fi

if [[ -f "Apptainer.def" ]]; then
    report_check "Apptainer.def exists" "pass"
else
    report_check "Apptainer.def exists" "fail" "File not found"
fi

if [[ -f "environment.yml" ]]; then
    report_check "environment.yml exists" "pass"
else
    report_check "environment.yml exists" "fail" "File not found"
fi

if [[ -f "requirements.txt" ]]; then
    report_check "requirements.txt exists" "pass"
else
    report_check "requirements.txt exists" "fail" "File not found"
fi

echo ""
echo "Validating Configuration:"
echo "------------------------"

# Check 2: Load and validate params.sh
if [[ -f "params.sh" ]]; then
    # Source params.sh in a subshell to avoid polluting current environment
    if bash -n params.sh 2>/dev/null; then
        report_check "params.sh syntax" "pass"
    else
        report_check "params.sh syntax" "fail" "Bash syntax error detected"
    fi

    # Check NAME variable
    source params.sh 2>/dev/null || true
    if [[ -n "${NAME:-}" ]]; then
        report_check "NAME variable set" "pass" "Container name: ${NAME}"
    else
        report_check "NAME variable set" "fail" "NAME not defined in params.sh"
    fi

    # Check deployment paths
    if [[ -n "${STAGE_PATH:-}" ]]; then
        report_check "STAGE_PATH configured" "pass"
    else
        report_check "STAGE_PATH configured" "warn" "Staging deployment path not set"
    fi

    if [[ -n "${DEPLOY_PATH:-}" ]]; then
        report_check "DEPLOY_PATH configured" "pass"
    else
        report_check "DEPLOY_PATH configured" "warn" "Production deployment path not set"
    fi
fi

echo ""
echo "Validating YAML Files:"
echo "---------------------"

# Check 3: Validate YAML syntax (if yamllint is available)
if command -v yamllint &> /dev/null; then
    if [[ -f "environment.yml" ]]; then
        if yamllint -d relaxed environment.yml > /dev/null 2>&1; then
            report_check "environment.yml syntax" "pass"
        else
            report_check "environment.yml syntax" "fail" "YAML syntax error detected"
        fi
    fi
else
    # Basic YAML validation using Python
    if command -v python3 &> /dev/null; then
        if [[ -f "environment.yml" ]]; then
            if python3 -c "import yaml; yaml.safe_load(open('environment.yml'))" 2>/dev/null; then
                report_check "environment.yml syntax" "pass"
            else
                report_check "environment.yml syntax" "fail" "YAML parsing error"
            fi
        fi
    else
        report_check "YAML validation" "warn" "yamllint and python3 not available - skipping YAML validation"
    fi
fi

echo ""
echo "Checking Environment Specification:"
echo "-----------------------------------"

# Check 4: Validate environment.yml content
if [[ -f "environment.yml" ]]; then
    if grep -q "name:" environment.yml; then
        ENV_NAME=$(grep "name:" environment.yml | head -n 1 | awk '{print $2}')
        if [[ "${ENV_NAME}" == "apptainer" ]]; then
            report_check "Conda environment name" "pass" "Environment name: ${ENV_NAME}"
        else
            report_check "Conda environment name" "warn" "Environment name is '${ENV_NAME}' (expected 'apptainer')"
        fi
    else
        report_check "Conda environment name" "fail" "No 'name:' field found in environment.yml"
    fi

    if grep -q "dependencies:" environment.yml; then
        report_check "Dependencies defined" "pass"
    else
        report_check "Dependencies defined" "fail" "No 'dependencies:' field found in environment.yml"
    fi
fi

echo ""
echo "Checking Git Status:"
echo "-------------------"

# Check 5: Git status
if git rev-parse --git-dir > /dev/null 2>&1; then
    CURRENT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    report_check "Git repository" "pass" "Current commit: ${CURRENT_COMMIT}"

    if [[ -n $(git status --porcelain) ]]; then
        report_check "Working directory clean" "warn" "Uncommitted changes detected - container will include this warning"
    else
        report_check "Working directory clean" "pass"
    fi

    # Check if current commit is tagged
    if git describe --exact-match --tags HEAD > /dev/null 2>&1; then
        GIT_TAG=$(git describe --exact-match --tags HEAD)
        report_check "Git tag" "pass" "Current commit tagged as: ${GIT_TAG}"
    fi
else
    report_check "Git repository" "warn" "Not in a git repository - version tracking will be limited"
fi

echo ""
echo "Checking Disk Space:"
echo "-------------------"

# Check 6: Disk space
AVAILABLE_GB=$(df -BG . | tail -n 1 | awk '{print $4}' | sed 's/G//')
if [[ ${AVAILABLE_GB} -ge 10 ]]; then
    report_check "Disk space" "pass" "${AVAILABLE_GB}GB available"
elif [[ ${AVAILABLE_GB} -ge 5 ]]; then
    report_check "Disk space" "warn" "Only ${AVAILABLE_GB}GB available (10GB+ recommended)"
else
    report_check "Disk space" "fail" "Only ${AVAILABLE_GB}GB available (insufficient for most builds)"
fi

# Final summary
echo ""
echo "========================================="
if [[ "${VALIDATION_OK}" == true ]]; then
    echo "✓ Validation passed! Ready to build."
    echo "========================================="
    exit 0
else
    echo "✗ Validation failed! Please fix errors before building."
    echo "========================================="
    exit 1
fi
