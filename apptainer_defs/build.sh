#!/bin/bash
# Build an Apptainer container from the definition file
# Usage: ./build.sh

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

# Check if Apptainer definition file exists
if [[ ! -f "Apptainer.def" ]]; then
    echo "Error: Apptainer.def not found in current directory" >&2
    exit 1
fi

# Check if apptainer command is available
if ! command -v apptainer &> /dev/null; then
    echo "Error: apptainer command not found. Please install Apptainer first." >&2
    exit 1
fi

# Gather version information
echo "Gathering version information..."

# Get git information
if git rev-parse --git-dir > /dev/null 2>&1; then
    # Use git describe --tags for automatic versioning
    # This gives: v1.0.0-5-g2414721 (tag + commits since tag + hash)
    # Or just: v1.0.0 if on a tagged commit
    # --always ensures we get at least the commit hash if no tags exist
    VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "unknown")

    # Strip leading 'v' if present for cleaner version numbers
    VERSION=${VERSION#v}

    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        GIT_DIRTY="true"
        GIT_STATUS="dirty (uncommitted changes)"
        # If version doesn't already have -dirty suffix, add it
        if [[ ! "${VERSION}" =~ -dirty$ ]]; then
            VERSION="${VERSION}-dirty"
        fi
    else
        GIT_DIRTY="false"
        GIT_STATUS="clean"
    fi

    # Check for git tag on current commit
    GIT_TAG=$(git describe --exact-match --tags HEAD 2>/dev/null || echo "")
else
    # Not in a git repository - fall back to VERSION file
    if [[ -f "VERSION" ]]; then
        VERSION=$(cat VERSION | tr -d '[:space:]')
        echo "Warning: Not in git repository, using VERSION file" >&2
    else
        VERSION="unknown"
        echo "Warning: Not in git repository and no VERSION file found" >&2
    fi
    GIT_COMMIT="not-in-git"
    GIT_BRANCH="unknown"
    GIT_DIRTY="unknown"
    GIT_STATUS="not in git repository"
    GIT_TAG=""
fi

# Create build metadata file
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILD_HOST=$(hostname)

# Write version info to a temporary file that will be included in the container
cat > .build_info <<EOF
VERSION=${VERSION}
GIT_COMMIT=${GIT_COMMIT}
GIT_BRANCH=${GIT_BRANCH}
GIT_DIRTY=${GIT_DIRTY}
GIT_TAG=${GIT_TAG}
BUILD_DATE=${BUILD_DATE}
BUILD_HOST=${BUILD_HOST}
CONTAINER_NAME=${NAME}
EOF

# Construct versioned container filename
CONTAINER_FILE="${NAME}-${VERSION}.sif"

echo "========================================="
echo "Building Apptainer container: ${CONTAINER_FILE}"
echo "========================================="
echo "Version: ${VERSION}"
echo "Git commit: ${GIT_COMMIT}"
if [[ -n "${GIT_TAG}" ]]; then
    echo "Git tag: ${GIT_TAG}"
fi
echo "Git branch: ${GIT_BRANCH}"
echo "Git status: ${GIT_STATUS}"
if [[ "${GIT_DIRTY}" == "true" ]]; then
    echo ""
    echo "⚠ WARNING: Building with uncommitted changes!"
    echo "   Container will be tagged as 'dirty'"
    echo ""
fi
echo "Build date: ${BUILD_DATE}"
echo "Start time: $(date)"
echo ""

# Export version variables for use in Apptainer.def
export BUILD_VERSION="${VERSION}"
export BUILD_GIT_COMMIT="${GIT_COMMIT}"
export BUILD_GIT_BRANCH="${GIT_BRANCH}"
export BUILD_GIT_DIRTY="${GIT_DIRTY}"
export BUILD_GIT_TAG="${GIT_TAG}"
export BUILD_DATE="${BUILD_DATE}"

# Build the container
if apptainer build "${CONTAINER_FILE}" Apptainer.def 2>&1 | tee build.log; then
    echo ""
    echo "========================================="
    echo "Build completed successfully!"
    echo "Container: ${CONTAINER_FILE}"
    echo "Log file: build.log"
    echo "End time: $(date)"
    echo ""
    echo "Container version information:"
    echo "  Version: ${VERSION}"
    echo "  Git commit: ${GIT_COMMIT}"
    if [[ -n "${GIT_TAG}" ]]; then
        echo "  Git tag: ${GIT_TAG}"
    fi
    if [[ "${GIT_DIRTY}" == "true" ]]; then
        echo "  Status: DIRTY (uncommitted changes)"
    fi
    echo ""
    echo "To inspect the container, run:"
    echo "  ./inspect.sh ${CONTAINER_FILE}"
    echo ""
    echo "To test the container, run:"
    echo "  ./test.sh ${CONTAINER_FILE}"
    echo "========================================="

    # Cleanup temporary build info file
    rm -f .build_info
else
    echo ""
    echo "=========================================" >&2
    echo "Build failed! Check build.log for details." >&2
    echo "End time: $(date)" >&2
    echo "=========================================" >&2

    # Cleanup temporary build info file
    rm -f .build_info
    exit 1
fi
