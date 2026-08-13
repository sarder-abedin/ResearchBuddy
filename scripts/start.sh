#!/usr/bin/env bash
# scripts/start.sh — BeeSearch universal launcher
#
# Detects your GPU (NVIDIA / AMD / Apple Silicon) and starts BeeSearch with
# the right Docker Compose configuration automatically.
# If LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are present in .env,
# the Langfuse observability stack starts alongside BeeSearch automatically.
#
# Usage:
#   ./scripts/start.sh           # start (uses cached image)
#   ./scripts/start.sh --build   # first run, or after pulling new code
#
# Press Ctrl-C to stop. The browser opens automatically when the app is ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

APP_URL="http://localhost:${PORT:-8000}"
LANGFUSE_URL="http://localhost:3000"
LANGFUSE_COMPOSE="$REPO_ROOT/docker-compose.langfuse.yml"

# ── Read a single key from .env without sourcing the file ─────────────────────

_read_dotenv_key() {
    local key="$1"
    local file="$REPO_ROOT/.env"
    if [[ -f "$file" ]]; then
        grep -E "^${key}=" "$file" | head -1 | sed 's/^[^=]*=//'
    fi
}

# ── Platform / GPU detection ──────────────────────────────────────────────────

detect_platform() {
    # macOS — native Ollama required (Docker can't run Ollama on Apple Silicon)
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "mac"
        return
    fi

    # NVIDIA — check nvidia-smi first, then fall back to the device node
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        echo "nvidia"
        return
    fi
    if [[ -e /dev/nvidia0 ]]; then
        echo "nvidia"
        return
    fi

    # AMD ROCm — /dev/kfd is the ROCm compute kernel device
    if [[ -e /dev/kfd ]]; then
        # Docker Desktop on Linux runs inside a VM and cannot pass /dev/kfd through.
        # In that case native Ollama (with ROCm) must run on the host instead.
        local ctx
        ctx=$(docker context show 2>/dev/null || echo "default")
        if [[ "$ctx" == "desktop-linux" ]]; then
            echo "amd-native"
        else
            echo "amd"
        fi
        return
    fi

    echo "cpu"
}

PLATFORM=$(detect_platform)

# ── Select compose files and print what was found ─────────────────────────────

case "$PLATFORM" in
    mac)
        echo "Detected: Apple Silicon Mac → native Ollama"
        if ! curl -sf http://localhost:11434 &>/dev/null; then
            echo ""
            echo "Ollama is not running. Please:"
            echo "  1. Install from https://ollama.com/download (macOS .dmg)"
            echo "  2. It starts automatically after installation"
            echo "  3. Pull the AI models once:"
            echo "       ollama pull llama3.2:3b"
            echo "       ollama pull nomic-embed-text"
            echo "  4. Run this script again"
            exit 1
        fi
        COMPOSE_CMD=(docker compose -f "$REPO_ROOT/docker-compose.mac.yml")
        ;;

    nvidia)
        echo "Detected: NVIDIA GPU → CUDA configuration"
        COMPOSE_CMD=(docker compose -f "$REPO_ROOT/docker-compose.yml" -f "$REPO_ROOT/docker-compose.gpu.yml")
        ;;

    amd)
        echo "Detected: AMD Radeon GPU (Docker Engine) → ROCm configuration"
        COMPOSE_CMD=(docker compose -f "$REPO_ROOT/docker-compose.yml" -f "$REPO_ROOT/docker-compose.gpu-amd.yml")
        ;;

    amd-native)
        echo "Detected: AMD Radeon GPU (Docker Desktop) → native Ollama required"
        if ! curl -sf http://localhost:11434 &>/dev/null; then
            echo ""
            echo "Native Ollama is not running. Please:"
            echo "  1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
            echo "     (ROCm is detected automatically if your AMD drivers are loaded)"
            echo "  2. Pull the AI models once:"
            echo "       ollama pull llama3.2:3b"
            echo "       ollama pull nomic-embed-text"
            echo "  3. Run this script again"
            exit 1
        fi
        COMPOSE_CMD=(docker compose -f "$REPO_ROOT/docker-compose.amd-native.yml")
        ;;

    cpu)
        echo "No GPU detected → CPU-only mode"
        COMPOSE_CMD=(docker compose -f "$REPO_ROOT/docker-compose.yml")
        ;;
esac

# ── Langfuse observability (optional) ────────────────────────────────────────
# If both keys are present in .env, start the Langfuse stack alongside BeeSearch.

LF_PK=$(_read_dotenv_key "LANGFUSE_PUBLIC_KEY")
LF_SK=$(_read_dotenv_key "LANGFUSE_SECRET_KEY")

if [[ -n "$LF_PK" && -n "$LF_SK" ]]; then
    LANGFUSE_ENABLED=1
    # Tell the web container to reach Langfuse by its Docker service name.
    # This must be exported before 'docker compose up' so the container picks it up.
    export LANGFUSE_HOST="http://beesearch-langfuse:3000"
else
    LANGFUSE_ENABLED=0
fi

# ── Shutdown on Ctrl-C ────────────────────────────────────────────────────────

LANGFUSE_STARTED=0

_cleanup() {
    echo ""
    echo "Shutting down…"
    "${COMPOSE_CMD[@]}" down --remove-orphans
    if [[ "$LANGFUSE_STARTED" == "1" ]]; then
        docker compose -f "$LANGFUSE_COMPOSE" down
    fi
}

trap _cleanup EXIT INT TERM

# ── Open browser once healthy ─────────────────────────────────────────────────

(
    echo "Waiting for BeeSearch at $APP_URL …"
    for i in $(seq 1 90); do
        if curl -sf "${APP_URL}/api/health" >/dev/null 2>&1; then
            echo ""
            echo "  BeeSearch → $APP_URL"
            if [[ "$LANGFUSE_ENABLED" == "1" ]]; then
                echo "  Langfuse  → $LANGFUSE_URL"
            fi
            echo ""
            if command -v xdg-open &>/dev/null; then
                xdg-open "$APP_URL"
            elif command -v open &>/dev/null; then
                open "$APP_URL"
            fi
            exit 0
        fi
        sleep 2
    done
    echo "App did not become ready within 180 s — open $APP_URL manually."
) &

# ── Launch ────────────────────────────────────────────────────────────────────

if [[ "$LANGFUSE_ENABLED" == "1" ]]; then
    echo ""
    echo "Langfuse keys found in .env — starting LLM observability stack…"
    # Start BeeSearch detached first — this creates the beesearch_default network
    # that docker-compose.langfuse.yml joins as an external network.
    "${COMPOSE_CMD[@]}" up -d "$@"
    # Start Langfuse now that the network exists
    docker compose -f "$LANGFUSE_COMPOSE" up -d
    LANGFUSE_STARTED=1
    # Follow logs in the foreground so Ctrl-C reaches _cleanup
    "${COMPOSE_CMD[@]}" logs -f
else
    "${COMPOSE_CMD[@]}" up "$@"
fi
