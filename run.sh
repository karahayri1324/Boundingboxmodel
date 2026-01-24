#!/bin/bash
# =============================================================================
# REAP V11 - Quick Start Script
# =============================================================================

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "  REAP V11 - Fixed VLM Expert Pruning"
    echo "=============================================="
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build      Build Docker image"
    echo "  shell      Interactive shell"
    echo "  reap       Run REAP scoring"
    echo "  prune      Prune model"
    echo "  test       Test pruned model"
    echo "  full       Run full pipeline"
    echo "  gpu        Show GPU status"
    echo "  clean      Remove container/image"
}

case "${1:-help}" in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        docker compose build
        ;;
    shell)
        echo -e "${GREEN}Starting shell...${NC}"
        docker compose run --rm reap bash
        ;;
    reap)
        echo -e "${GREEN}Running REAP scoring...${NC}"
        docker compose run --rm reap python reap.py
        ;;
    prune)
        echo -e "${GREEN}Pruning model...${NC}"
        docker compose run --rm reap python prune_model.py --verify
        ;;
    test)
        echo -e "${GREEN}Testing model...${NC}"
        docker compose run --rm reap python test_pruned_model.py
        ;;
    full)
        echo -e "${GREEN}Running full pipeline...${NC}"
        echo -e "${BLUE}[1/3] REAP Scoring${NC}"
        docker compose run --rm reap python reap.py
        echo -e "${BLUE}[2/3] Pruning${NC}"
        docker compose run --rm reap python prune_model.py --verify
        echo -e "${BLUE}[3/3] Testing${NC}"
        docker compose run --rm reap python test_pruned_model.py
        echo -e "${GREEN}Done!${NC}"
        ;;
    gpu)
        nvidia-smi
        ;;
    clean)
        echo -e "${RED}Cleaning...${NC}"
        docker compose down --rmi local -v
        ;;
    help|--help|-h)
        print_header
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown: $1${NC}"
        print_usage
        exit 1
        ;;
esac
