#!/bin/bash
# Inspect an Apptainer container and show detailed information
# Usage: ./inspect.sh [container.sif]

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Load configuration
if [[ -f "params.sh" ]]; then
    source params.sh
fi

# Determine container file
if [[ $# -gt 0 ]]; then
    CONTAINER_FILE="$1"
elif [[ -n "${NAME:-}" ]]; then
    CONTAINER_FILE="${NAME}.sif"
else
    CONTAINER_FILE="container.sif"
fi

# Check if container exists
if [[ ! -f "${CONTAINER_FILE}" ]]; then
    echo "Error: Container file '${CONTAINER_FILE}' not found" >&2
    echo "" >&2
    echo "Usage: $0 [container.sif]" >&2
    echo "  If no file specified, uses NAME from params.sh" >&2
    exit 1
fi

# Check if apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "Error: apptainer command not found" >&2
    exit 1
fi

echo "========================================="
echo "Inspecting Container: ${CONTAINER_FILE}"
echo "========================================="
echo ""

# Basic file information
echo "File Information:"
echo "----------------"
FILE_SIZE=$(du -h "${CONTAINER_FILE}" | cut -f1)
FILE_DATE=$(stat -c %y "${CONTAINER_FILE}" 2>/dev/null || stat -f "%Sm" "${CONTAINER_FILE}" 2>/dev/null || echo "unknown")
echo "Size: ${FILE_SIZE}"
echo "Created/Modified: ${FILE_DATE}"
echo ""

# Container labels
echo "Container Labels:"
echo "----------------"
if apptainer inspect --labels "${CONTAINER_FILE}" 2>/dev/null; then
    echo ""
else
    echo "No labels found or unable to read labels"
    echo ""
fi

# Definition file (if embedded)
echo "Definition Information:"
echo "----------------------"
if apptainer inspect --deffile "${CONTAINER_FILE}" 2>/dev/null | head -n 20; then
    echo "... (showing first 20 lines, use 'apptainer inspect --deffile' for full definition)"
    echo ""
else
    echo "Definition file not embedded in container"
    echo ""
fi

# Environment variables
echo "Environment Variables:"
echo "---------------------"
if apptainer inspect --environment "${CONTAINER_FILE}" 2>/dev/null; then
    echo ""
else
    echo "No environment variables found"
    echo ""
fi

# Runscript
echo "Runscript:"
echo "---------"
if apptainer inspect --runscript "${CONTAINER_FILE}" 2>/dev/null; then
    echo ""
else
    echo "No runscript defined"
    echo ""
fi

# Help text
echo "Container Help:"
echo "--------------"
if apptainer run-help "${CONTAINER_FILE}" 2>/dev/null; then
    echo ""
else
    echo "No help text available"
    echo ""
fi

# Python and package information (if available)
echo "Python Environment:"
echo "------------------"
if apptainer exec "${CONTAINER_FILE}" which python &> /dev/null; then
    PYTHON_PATH=$(apptainer exec "${CONTAINER_FILE}" which python 2>/dev/null || echo "unknown")
    PYTHON_VERSION=$(apptainer exec "${CONTAINER_FILE}" python --version 2>&1 || echo "unknown")
    echo "Python path: ${PYTHON_PATH}"
    echo "Python version: ${PYTHON_VERSION}"

    # Check for conda
    if apptainer exec "${CONTAINER_FILE}" bash -c 'echo $CONDA_DEFAULT_ENV' 2>/dev/null | grep -q "apptainer"; then
        echo "Conda environment: apptainer (activated)"
    fi

    echo ""

    # Key packages
    echo "Key Packages:"
    echo "------------"

    # Check for JupyterLab
    if apptainer exec "${CONTAINER_FILE}" which jupyter-lab &> /dev/null; then
        JUPYTER_VERSION=$(apptainer exec "${CONTAINER_FILE}" jupyter-lab --version 2>&1 || echo "unknown")
        echo "JupyterLab: ${JUPYTER_VERSION}"
    fi

    # Check for PyTorch
    if apptainer exec "${CONTAINER_FILE}" python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        TORCH_INFO=$(apptainer exec "${CONTAINER_FILE}" python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch info unavailable")
        echo "${TORCH_INFO}"
    fi

    # Check for TensorFlow
    if apptainer exec "${CONTAINER_FILE}" python -c "import tensorflow" 2>/dev/null; then
        TF_VERSION=$(apptainer exec "${CONTAINER_FILE}" python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>/dev/null || echo "unknown")
        echo "${TF_VERSION}"
    fi

    # Check for common data science packages
    echo ""
    echo "Data Science Packages:"
    for pkg in numpy pandas scikit-learn matplotlib seaborn scipy; do
        if apptainer exec "${CONTAINER_FILE}" python -c "import ${pkg}; print(f'{${pkg}.__name__} {${pkg}.__version__}')" 2>/dev/null; then
            VERSION=$(apptainer exec "${CONTAINER_FILE}" python -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "unknown")
            echo "  ${pkg}: ${VERSION}"
        fi
    done

else
    echo "Python not found in container"
fi

echo ""
echo "========================================="
echo "Inspection Complete"
echo "========================================="
echo ""
echo "For more detailed information, use:"
echo "  apptainer inspect ${CONTAINER_FILE}"
echo "  apptainer exec ${CONTAINER_FILE} conda list"
echo "  apptainer exec ${CONTAINER_FILE} pip list"
