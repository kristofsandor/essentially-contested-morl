#!/bin/bash
# Stage the Apptainer container to testing/staging location
# Usage: ./stage.sh [container.sif]
#   If no container specified, finds the most recent ${NAME}-*.sif file

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

# Determine which container to stage
if [[ $# -gt 0 ]]; then
    CONTAINER_FILE="$1"
else
    # Find the most recently modified container file matching NAME-*.sif
    CONTAINER_FILE=$(ls -t ${NAME}-*.sif 2>/dev/null | head -n 1 || echo "")
    if [[ -z "${CONTAINER_FILE}" ]]; then
        echo "Error: No container file ${NAME}-*.sif found" >&2
        echo "Please run build.sh first to create the container" >&2
        exit 1
    fi
    echo "Using most recent container: ${CONTAINER_FILE}"
fi

# Check if container file exists
if [[ ! -f "${CONTAINER_FILE}" ]]; then
    echo "Error: Container file ${CONTAINER_FILE} not found" >&2
    exit 1
fi

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "Error: rsync command not found. Please install rsync first." >&2
    exit 1
fi

echo "========================================="
echo "Staging container: ${CONTAINER_FILE}"
echo "========================================="
echo ""

# Stage to testing environment
# Note: Update STAGE_PATH in params.sh to match your staging location
# The destination assumes you have SSH access and the same username on the remote system

# Validate STAGE_PATH is set
if [[ -z "${STAGE_PATH:-}" ]]; then
    echo "Error: STAGE_PATH not set in params.sh" >&2
    exit 1
fi

# Expand variables in STAGE_PATH and append container filename
DEST_DIR=$(eval echo "${STAGE_PATH}")
DEST_PATH="${DEST_DIR}${CONTAINER_FILE}"

echo "Staging to: ${DEST_PATH}"
echo ""

if rsync -avc --progress "${CONTAINER_FILE}" "${DEST_PATH}"; then
    echo ""
    echo "========================================="
    echo "Staging completed successfully!"
    echo "Container staged to: ${DEST_PATH}"
    echo "========================================="
else
    echo ""
    echo "=========================================" >&2
    echo "Staging failed!" >&2
    echo "Please check:"
    echo "  - SSH connection to remote host"
    echo "  - Permissions on destination directory"
    echo "  - Available disk space on remote host"
    echo "=========================================" >&2
    exit 1
fi

