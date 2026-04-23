# Apptainer Container Template

A minimal, fast-building template with Python 3.11 + JupyterLab and automated git-based versioning.

## Overview

This template provides a complete workflow for building, testing, staging, and deploying Apptainer containers. It includes:

- **Minimal base**: Python 3.11 + JupyterLab only
- **Fast builds**: No heavy packages - add only what you need
- **Auto-activated conda environment** in JupyterLab terminals and shells
- **Git-based versioning** - Automatic version tracking via `git describe --tags`
- **Versioned container files** - Each build creates `name-version.sif`
- Automated build, test, and deployment scripts
- Comprehensive validation and dependency checking
- Container inspection and metadata tools
- Makefile for convenient operations
- Commented examples for common packages (PyTorch, data science, etc.)

**Philosophy**: Start minimal and fast, add packages as needed. Perfect for quick iteration and testing.

## Quick Start

### Using Make (Recommended)
```bash
# Check dependencies and validate configuration
make check validate

# Build the container (creates versioned .sif file)
make build

# Test the container
make test

# Inspect the container
make inspect

# Deploy (automatically uses most recent build)
make stage   # Deploy to staging
make deploy  # Deploy to production
```

### Using Scripts Directly

1. **Check system dependencies**:
   ```bash
   ./check-dependencies.sh
   ```

2. **Configure your container** - Edit `params.sh`:
   ```bash
   export NAME=my-container
   export STAGE_PATH="user@host:/path/to/staging/"
   export DEPLOY_PATH="user@host:/path/to/production/"
   ```

   **Note:** STAGE_PATH and DEPLOY_PATH should be directories (ending with `/`).
   The versioned container filename will be appended automatically.

3. **Add packages (optional)** - Uncomment packages in `environment.yml` and `requirements.txt`

   Default build includes only Python 3.11 + JupyterLab for fast testing

4. **Validate configuration**:
   ```bash
   ./validate.sh
   ```

5. **Build the container**:
   ```bash
   ./build.sh
   ```
   Creates a versioned container file (e.g., `template-1.0.0-5-g53f3e09.sif`)

   Version is automatically generated from git using `git describe --tags`

6. **Test the container**:
   ```bash
   ./test.sh
   ```

7. **Inspect the container**:
   ```bash
   ./inspect.sh
   ```

8. **Deploy**:
   ```bash
   ./stage.sh   # Deploy to staging/testing
   ./deploy.sh  # Deploy to production
   ```

## File Structure

```
template/
├── Apptainer.def           # Container definition file
├── params.sh               # Configuration parameters
├── environment.yml         # Conda environment specification
├── requirements.txt        # Python pip requirements
├── Makefile               # Convenient make targets
├── build.sh               # Build container with version tracking
├── test.sh                # Test container functionality
├── validate.sh            # Validate configuration before build
├── check-dependencies.sh  # Check system requirements
├── inspect.sh             # Inspect built container
├── stage.sh               # Deploy to staging
├── deploy.sh              # Deploy to production
├── clean.sh               # Remove build artifacts
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## Scripts

### Makefile
Convenient interface for all operations.

```bash
make help       # Show all available targets
make check      # Check system dependencies
make validate   # Validate configuration
make build      # Build container
make test       # Test container
make inspect    # Inspect container
make stage      # Deploy to staging
make deploy     # Deploy to production
make clean      # Remove artifacts
make all        # Check, validate, and build
```

### check-dependencies.sh
Verifies all required system dependencies are installed.

Checks for:
- apptainer, rsync, git, bash
- Disk space availability
- Git repository status
- Optional tools (make, yamllint)

### validate.sh
Pre-flight validation before building.

Validates:
- Required files exist (params.sh, Apptainer.def, etc.)
- Configuration syntax (bash, YAML)
- Environment specification
- Git status and uncommitted changes
- Disk space requirements

### build.sh
Builds the Apptainer container with version tracking.

Features:
- Reads VERSION file
- Captures git commit hash and branch
- Detects uncommitted changes (warns if dirty)
- Embeds version metadata in container labels
- Validates all required files exist
- Provides detailed progress output
- Logs build output to `build.log`

The container will include:
- Version number from VERSION file
- Git commit hash
- Git branch name
- "Dirty" flag if uncommitted changes exist
- Build timestamp
- Build hostname

### test.sh
Runs comprehensive tests on the built container.

Tests include:
- Python availability and version
- Conda environment activation
- JupyterLab installation and version
- Environment export for verification

Returns exit code 0 if all tests pass, 1 if any fail.

### inspect.sh
Displays detailed information about a built container.

Shows:
- File size and modification date
- Container labels (including version info)
- Definition file (first 20 lines)
- Environment variables
- Runscript
- Help text
- Python environment details
- Installed packages (PyTorch, TensorFlow, data science libraries)

Usage:
```bash
./inspect.sh                # Uses NAME from params.sh
./inspect.sh container.sif  # Inspect specific container
```

### stage.sh
Deploys container to staging/testing environment using rsync.

Usage:
```bash
./stage.sh                     # Auto-deploys most recent build
./stage.sh container-1.0.0.sif # Deploy specific version
```

Configure `STAGE_PATH` in `params.sh` before running.

### deploy.sh
Deploys container to production environment using rsync.

Usage:
```bash
./deploy.sh                    # Auto-deploys most recent build
./deploy.sh container-1.0.0.sif # Deploy specific version
```

Configure `DEPLOY_PATH` in `params.sh` before running.

**Note:** When no filename is specified, both scripts automatically find and deploy the most recently built container matching `${NAME}-*.sif`.

### clean.sh
Removes build artifacts and temporary files.

```bash
./clean.sh        # Remove logs and temp files
./clean.sh --all  # Remove container images too
```

## Configuration

### params.sh
Central configuration file for all scripts.

Required variables:
- `NAME` - Container name (used for .sif filename)
- `STAGE_PATH` - Staging deployment destination
- `DEPLOY_PATH` - Production deployment destination

### environment.yml
Conda environment specification with the `apptainer` environment.

**Minimal base (installed by default):**
- Python 3.11
- JupyterLab
- pip

**Auto-activation:** The `apptainer` conda environment is automatically activated in:
- Apptainer shell sessions
- Apptainer exec commands
- JupyterLab terminal windows

**Adding packages:** Uncomment examples or add your own with version pins for reproducibility:
```yaml
dependencies:
  - numpy==1.26.4
  - pandas==2.2.0
  - scikit-learn==1.4.0
```

### requirements.txt
Python packages installed via pip.

**Default:** Empty - all packages are commented examples

**Adding packages:** Uncomment what you need or add your own:
```txt
tqdm==4.66.1
plotly==5.18.0
```

For PyTorch with CUDA, uncomment and adjust CUDA version. See: https://pytorch.org/get-started/locally/

## Apptainer Definition File

The `Apptainer.def` file defines the container build process:

### Key Sections

- **%labels** - Metadata (author, version, description)
- **%help** - Usage instructions (accessible via `apptainer run-help`)
- **%files** - Files copied from host to container
- **%post** - Build-time commands (package installation, setup)
- **%environment** - Runtime environment variables
- **%runscript** - Default action when container is run

### Base Image
Uses `condaforge/miniforge3:latest` which includes:
- Minimal conda installation (mamba)
- Python ecosystem tools
- Debian-based system

## Included Packages

### Minimal Base
- **Python 3.11**
- **JupyterLab** - Interactive notebooks and development environment
- **pip** - Package installer

### System Packages
- **build-essential** - Compilers for building Python packages
- **git** - Version control
- **vim** - Text editor

### Ready to Add (commented examples in config files)
All packages are commented out by default. Uncomment what you need:

**Data Science**: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy

**Deep Learning**: PyTorch with CUDA, TensorFlow, transformers

**Utilities**: tqdm, plotly, requests, ipywidgets

**Development**: black, pytest, mypy, jupyterlab-git

## Usage Examples

### Interactive Shell
```bash
# Use the versioned container filename
apptainer shell template-1.0.0.sif

# Conda 'apptainer' environment is auto-activated
(apptainer) $ python --version
(apptainer) $ jupyter --version
```

### Execute Python Script
```bash
apptainer exec template-1.0.0.sif python script.py
```

### Run JupyterLab
```bash
apptainer exec template-1.0.0.sif jupyter-lab --no-browser --ip=0.0.0.0
# Open terminals in JupyterLab - 'apptainer' env is auto-activated!
```

### Check Installed Packages
```bash
apptainer exec template-1.0.0.sif pip list
apptainer exec template-1.0.0.sif conda list
```

### Check Container Info
```bash
apptainer inspect template-1.0.0.sif
apptainer run-help template-1.0.0.sif
./inspect.sh template-1.0.0.sif  # Detailed inspection
```

## Version Tracking and Git Integration

This template uses **git-based versioning** - no manual version files to maintain!

### How It Works

When you run `./build.sh` or `make build`, the build system automatically:

1. **Generates version** using `git describe --tags --always --dirty`:
   - `1.0.0` - On exact tag `v1.0.0`
   - `1.0.0-5-g2414721` - 5 commits after tag `v1.0.0`
   - `1.0.0-dirty` - On tag with uncommitted changes
   - `53f3e09` - No tags yet (commit hash only)
   - `53f3e09-dirty` - No tags, with uncommitted changes

2. **Captures metadata**:
   - Git commit hash and branch
   - Build timestamp and hostname
   - Dirty flag if uncommitted changes exist

3. **Embeds in container**:
   - Stored in `/.build_info` file
   - Displayed in runscript output
   - Shown during build

**Every commit gets a unique, traceable version automatically!**

### Versioned Container Filenames

Built containers include the version in their filename:
- `template-1.0.0.sif` - Release version
- `template-1.0.0-5-g2414721.sif` - Development version
- `template-53f3e09-dirty.sif` - Uncommitted changes

This makes it easy to:
- Keep multiple versions side-by-side
- Identify container versions at a glance
- Track which version is deployed where

### Version Information in Containers

The version metadata is stored in:

**Container Labels**: Accessible via `apptainer inspect --labels`
```bash
$ apptainer inspect --labels my-container.sif
org.opencontainers.image.version: 1.0.0
org.opencontainers.image.revision: abc123f
git.commit: abc123f
git.branch: main
git.dirty: false
```

**Build Info File**: Stored in container at `/.build_info`
```bash
$ apptainer exec my-container.sif cat /.build_info
VERSION=1.0.0
GIT_COMMIT=abc123f
GIT_BRANCH=main
GIT_DIRTY=false
BUILD_DATE=2024-01-15T10:30:00Z
```

**Runscript Output**: Displayed when running the container
```bash
$ apptainer run my-container.sif
=========================================
Apptainer Container
=========================================
Version: 1.0.0
Git commit: abc123f
Built: 2024-01-15T10:30:00Z
...
```

### Working with Uncommitted Changes

If you build with uncommitted changes, the system will:
- Display a warning during build
- Set `git.dirty: true` in container labels
- Show a warning when running the container

This helps you track whether a container was built from a clean git state or contains experimental changes.

**Best Practice**: Always commit your changes before building production containers.

### Tagging Releases

Use git tags to mark stable releases following semantic versioning:
- **MAJOR** (v1.x.x): Breaking changes
- **MINOR** (vx.1.x): New features, backwards compatible
- **PATCH** (vx.x.1): Bug fixes

**Development workflow:**
```bash
# Make changes to environment.yml
vim environment.yml

# Commit changes
git add environment.yml
git commit -m "Add new packages for data processing"

# Build (automatically gets version like: 1.0.0-1-g53f3e09)
make build

# Container version is: 1.0.0-1-g53f3e09
# Meaning: 1 commit after tag v1.0.0, at commit g53f3e09
```

**Release workflow:**
```bash
# When ready to release
git tag -a v1.1.0 -m "Release 1.1.0: Added data processing packages"

# Build (automatically gets clean version: 1.1.0)
make build

# Container version is: 1.1.0
# This is a tagged release!

# Push tag to remote
git push origin v1.1.0
```

**Every commit gets a unique version** - no manual version file updates needed!

## Best Practices

1. **Tag releases** - Use git tags (`v1.0.0`) for production releases - version is automatic!
2. **Commit before building** - Build production containers from clean git state (avoid `-dirty`)
3. **Pin package versions** - For reproducibility in production (add `==version` to dependencies)
4. **Validate before building** - Run `make validate` to catch issues early
5. **Test thoroughly** - Run `make test` before deploying to production
6. **Use staging environment** - Test in staging (`make stage`) before production deploy
7. **Inspect containers** - Use `./inspect.sh` to verify packages and versions
8. **Document changes** - Add comments explaining why packages are needed
9. **Use the Makefile** - `make all` runs checks, validation, and build automatically
10. **Monitor CUDA versions** - Update PyTorch CUDA version in `requirements.txt` as needed
11. **Leverage auto-versioning** - Every commit gets a unique version, no manual updates needed

## Troubleshooting

### Build fails
- Check `build.log` for detailed error messages
- Ensure all required files (environment.yml, requirements.txt) exist
- Verify apptainer is installed and accessible

### Deployment fails
- Check SSH access to remote host
- Verify destination directory exists and is writable
- Ensure sufficient disk space on remote system

### Tests fail
- Container may not have been built successfully
- Check that required packages are listed in environment.yml
- Run `./build.sh` again to rebuild

## Support

For issues or questions:
- Check the logs in `build.log` and test output
- Verify all configuration in `params.sh`
- Review Apptainer documentation: https://apptainer.org/docs/

## License

Update this section with your project's license information.
