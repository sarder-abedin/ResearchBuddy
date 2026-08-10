#!/usr/bin/env bash
# scripts/start.sh — BeeSearch universal launcher
#
# Detects your GPU (NVIDIA / AMD / Apple Silicon) and starts BeeSearch with
# the right Docker Compose configuration automatically.
#
# Usage:
#   ./scripts/start.sh           # start (uses cached image)
#   ./scripts/start.sh --build   # first run, or after pulling new code
#
# Press Ctrl-C to stop. The browser opens automatically when the app is ready.

set -euo pipefail

APP_URL="http://localhost:${PORT:-8000}"

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
        COMPOSE_CMD=(docker compose -f docker-compose.mac.yml)
        ;;

    nvidia)
        echo "Detected: NVIDIA GPU → CUDA configuration"
        COMPOSE_CMD=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml)
        ;;

    amd)
        echo "Detected: AMD Radeon GPU (Docker Engine) → ROCm configuration"
        COMPOSE_CMD=(docker compose -f docker-compose.yml -f docker-compose.gpu-amd.yml)
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
        COMPOSE_CMD=(docker compose -f docker-compose.amd-native.yml)
        ;;

    cpu)
        echo "No GPU detected → CPU-only mode"
        COMPOSE_CMD=(docker compose -f docker-compose.yml)
        ;;
esac

# ── Shutdown on Ctrl-C ────────────────────────────────────────────────────────

_cleanup() {
    echo ""
    echo "Shutting down…"
    "${COMPOSE_CMD[@]}" down --remove-orphans
}

trap _cleanup EXIT INT TERM

# ── Open browser once healthy ─────────────────────────────────────────────────

(
    echo "Waiting for BeeSearch at $APP_URL …"
    for i in $(seq 1 90); do
        if curl -sf "${APP_URL}/api/health" >/dev/null 2>&1; then
            echo ""
            echo "BeeSearch is ready — opening $APP_URL"
            if command -v xdg-open &>/dev/null; then
                xdg-open "$APP_URL"
            elif command -v open &>/dev/null; then
                open "$APP_URL"
            else
                echo "Open your browser at: $APP_URL"
            fi
            exit 0
        fi
        sleep 2
    done
    echo "App did not become ready within 180 s — open $APP_URL manually."
) &

# ── Launch ────────────────────────────────────────────────────────────────────

"${COMPOSE_CMD[@]}" up "$@"
