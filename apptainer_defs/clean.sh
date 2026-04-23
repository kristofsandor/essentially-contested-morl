#!/bin/bash
# Clean up build artifacts and temporary files
# Usage: ./clean.sh [--all]

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Parse command line arguments
CLEAN_ALL=false
if [[ "${1:-}" == "--all" ]]; then
    CLEAN_ALL=true
fi

echo "========================================="
echo "Cleaning build artifacts"
echo "========================================="
echo ""

# Load configuration to get NAME variable
if [[ -f "params.sh" ]]; then
    source params.sh
else
    echo "Warning: params.sh not found, using default pattern for cleanup"
    NAME="*"
fi

# Function to safely remove files
safe_remove() {
    local pattern=$1
    local description=$2

    if compgen -G "${pattern}" > /dev/null; then
        echo "Removing ${description}..."
        rm -v ${pattern}
    else
        echo "No ${description} found."
    fi
}

# Remove container images
if [[ "${CLEAN_ALL}" == true ]]; then
    safe_remove "*.sif" "container images (*.sif)"
elif [[ "${NAME}" != "*" ]]; then
    safe_remove "${NAME}.sif" "container image (${NAME}.sif)"
fi

# Remove log files
safe_remove "build.log" "build log"
safe_remove "slurm-*.out" "SLURM output logs"
safe_remove "slurm-*.err" "SLURM error logs"

# Remove exported environment files
safe_remove "environment_export.yml" "exported environment file"

# Remove temporary files
safe_remove "*.tmp" "temporary files"
safe_remove "*.bak" "backup files"
safe_remove "*~" "editor backup files"

echo ""
echo "========================================="
echo "Cleanup completed!"
echo "========================================="

if [[ "${CLEAN_ALL}" == false && -f "${NAME}.sif" ]]; then
    echo ""
    echo "Note: To remove ALL container images, run:"
    echo "  ./clean.sh --all"
fi
