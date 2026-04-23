#!/bin/bash
# Configuration parameters for Apptainer container build and deployment
# This file is sourced by other scripts in this directory

# Container name (will be used for the .sif file name)
# Change this to match your specific use case
export NAME=template

# Staging environment deployment directory
# Update this to match your staging/testing server and path
# Format: [user@]host:path or local_path (directory only, filename will be added automatically)
export STAGE_PATH="dblue:/home/${USER}/ondemand/jupyter/"

# Production deployment directory
# Update this to match your production server and path
# Format: [user@]host:path or local_path (directory only, filename will be added automatically)
export DEPLOY_PATH="daic:/tudelft.net/staff-umbrella/reit/apptainer/"

# Validate that NAME is not empty
if [[ -z "${NAME}" ]]; then
    echo "Error: NAME must not be empty" >&2
    return 1 2>/dev/null || exit 1
fi

# Validate that deployment paths are set
if [[ -z "${STAGE_PATH}" ]]; then
    echo "Warning: STAGE_PATH is not set in params.sh" >&2
fi

if [[ -z "${DEPLOY_PATH}" ]]; then
    echo "Warning: DEPLOY_PATH is not set in params.sh" >&2
fi

