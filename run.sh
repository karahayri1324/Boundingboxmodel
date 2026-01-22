#!/bin/bash
# =============================================================================
# REAP V10 - Quick Start Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "  REAP V10 - Expert Pruning for Qwen3-VL"
    echo "  H200 Optimized"
    echo "=============================================="
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build      Build Docker image"
    echo "  shell      Enter Docker shell (interactive)"
    echo "  reap       Run REAP scoring"
    echo "  prune      Prune model after scoring"
    echo "  test       Test pruned model"
    echo "  full       Run full pipeline (reap -> prune -> test)"
    echo "  logs       Show container logs"
    echo "  stop       Stop container"
    echo "  clean      Remove container and image"
    echo "  gpu        Show GPU status"
    echo ""
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not found!${NC}"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker daemon not running!${NC}"
        exit 1
    fi
}

check_nvidia() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
        return 1
    fi

    if ! nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: NVIDIA driver issue${NC}"
        return 1
    fi

    return 0
}

cmd_build() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker compose build
    echo -e "${GREEN}Build complete!${NC}"
}

cmd_shell() {
    echo -e "${GREEN}Starting interactive shell...${NC}"
    docker compose run --rm reap bash
}

cmd_reap() {
    echo -e "${GREEN}Running REAP scoring...${NC}"
    docker compose run --rm reap python reap.py
}

cmd_prune() {
    echo -e "${GREEN}Pruning model...${NC}"
    docker compose run --rm reap python prune_model.py --verify
}

cmd_test() {
    echo -e "${GREEN}Testing pruned model...${NC}"
    docker compose run --rm reap python test_reapv8_docker.py
}

cmd_full() {
    echo -e "${GREEN}Running full pipeline...${NC}"
    echo ""

    echo -e "${BLUE}[1/3] REAP Scoring${NC}"
    cmd_reap

    echo ""
    echo -e "${BLUE}[2/3] Model Pruning${NC}"
    cmd_prune

    echo ""
    echo -e "${BLUE}[3/3] Testing${NC}"
    cmd_test

    echo ""
    echo -e "${GREEN}Pipeline complete!${NC}"
}

cmd_logs() {
    docker compose logs -f reap
}

cmd_stop() {
    echo -e "${YELLOW}Stopping container...${NC}"
    docker compose down
}

cmd_clean() {
    echo -e "${RED}Removing container and image...${NC}"
    docker compose down --rmi local -v
}

cmd_gpu() {
    if check_nvidia; then
        echo -e "${GREEN}GPU Status:${NC}"
        nvidia-smi
    fi
}

# =============================================================================
# Main
# =============================================================================

print_header
check_docker

case "${1:-help}" in
    build)
        cmd_build
        ;;
    shell)
        cmd_shell
        ;;
    reap)
        cmd_reap
        ;;
    prune)
        cmd_prune
        ;;
    test)
        cmd_test
        ;;
    full)
        cmd_full
        ;;
    logs)
        cmd_logs
        ;;
    stop)
        cmd_stop
        ;;
    clean)
        cmd_clean
        ;;
    gpu)
        cmd_gpu
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
