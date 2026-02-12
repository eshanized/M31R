#!/usr/bin/env bash
# ==============================================================================
# M31R Release Automation: HuggingFace Uploader
# ==============================================================================
#
# Automates uploading release artifacts to HuggingFace Hub.
# Strictly idempotent, safe, and non-interactive.
#
# Usage:
#   snigdhaos-upload-hf.sh <release_dir>
#
# Requirements:
#   - huggingface-cli
#   - HF_TOKEN env var
#   - HF_REPO_ID env var (e.g., "my-org/m31r-tiny")
#
# Author: Eshan Roy
# Status: PROD
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
RELEASE_DIR="${1:-}"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
log_info() { echo -e "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo -e "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2; }

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
    log_error "HF_TOKEN environment variable is not set."
    exit 1
fi

if [[ -z "${HF_REPO_ID:-}" ]]; then
    log_error "HF_REPO_ID environment variable is not set."
    exit 1
fi

if [[ -z "${RELEASE_DIR}" ]]; then
    log_error "Usage: $0 <release_dir>"
    exit 1
fi

if [[ ! -d "${RELEASE_DIR}" ]]; then
    log_error "Release directory not found: ${RELEASE_DIR}"
    exit 1
fi

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

log_info "Starting upload process for: ${RELEASE_DIR}"
log_info "Target Repository: ${HF_REPO_ID}"

# 1. Login (non-interactive)
log_info "Authenticating..."
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

# 2. Ensure Repo Exists
log_info "Ensuring repository exists..."
# || true allows it to fail if repo exists
huggingface-cli repo create "${HF_REPO_ID}" --type model || log_info "Repo likely exists, proceeding."

# 3. Upload Artifacts
# We upload the CONTENTS of the release dir to the ROOT of the repo.
log_info "Uploading artifacts..."
huggingface-cli upload \
    "${HF_REPO_ID}" \
    "${RELEASE_DIR}" \
    . \
    --repo-type model \
    --commit-message "Release: $(basename "${RELEASE_DIR}")"

log_info "Upload complete."
exit 0
