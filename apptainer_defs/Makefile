# Makefile for Apptainer container management
#
# Usage:
#   make help       - Show this help message
#   make check      - Check system dependencies
#   make validate   - Validate configuration files
#   make build      - Build the container
#   make test       - Test the container
#   make inspect    - Inspect the built container
#   make stage      - Deploy to staging environment
#   make deploy     - Deploy to production
#   make clean      - Remove build artifacts
#   make clean-all  - Remove all artifacts including container
#   make all        - Run check, validate, and build

.PHONY: help check validate build test inspect stage deploy clean clean-all all

# Default target
.DEFAULT_GOAL := help

help:
	@echo "========================================="
	@echo "Apptainer Container Management"
	@echo "========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make help       - Show this help message"
	@echo "  make check      - Check system dependencies"
	@echo "  make validate   - Validate configuration files"
	@echo "  make build      - Build the container"
	@echo "  make test       - Test the container"
	@echo "  make inspect    - Inspect the built container"
	@echo "  make stage      - Deploy to staging environment"
	@echo "  make deploy     - Deploy to production"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make clean-all  - Remove all artifacts including container"
	@echo "  make all        - Run check, validate, and build"
	@echo ""
	@echo "Workflow examples:"
	@echo "  make all test          - Build and test"
	@echo "  make build test stage  - Build, test, and stage"
	@echo ""

check:
	@echo "Checking system dependencies..."
	@./check-dependencies.sh

validate:
	@echo "Validating configuration..."
	@./validate.sh

build: validate
	@echo "Building container..."
	@./build.sh

test:
	@echo "Testing container..."
	@./test.sh

inspect:
	@echo "Inspecting container..."
	@./inspect.sh

stage:
	@echo "Deploying to staging..."
	@./stage.sh

deploy:
	@echo "Deploying to production..."
	@./deploy.sh

clean:
	@echo "Cleaning build artifacts..."
	@./clean.sh

clean-all:
	@echo "Cleaning all artifacts including container..."
	@./clean.sh --all

all: check validate build
	@echo ""
	@echo "========================================="
	@echo "Build complete! Next steps:"
	@echo "  make test    - Test the container"
	@echo "  make inspect - Inspect the container"
	@echo "  make stage   - Deploy to staging"
	@echo "========================================="
