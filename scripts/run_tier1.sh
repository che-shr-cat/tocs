#!/usr/bin/env bash
# Run Tier 1 experiments: APG (1A) + Probe Ablation (1B)
# Model: openai/gpt-5.3-codex
# Codebases: 3 seeds (42, 123, 999)
#
# Total runs: 15
#   Tier 1A (APG): 9 runs (3 passive modes × 3 codebases)
#   Tier 1B (Probe Ablation): 6 runs (2 tracks × 3 codebases)
#
# Estimated cost: ~$22-27 total
#
# Usage:
#   ./scripts/run_tier1.sh           # Run all
#   ./scripts/run_tier1.sh 1a        # Run APG only
#   ./scripts/run_tier1.sh 1b        # Run probe ablation only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON=".venv/bin/python"
MODEL="openai/gpt-5.3-codex"
BUDGET=20
PROBE_INTERVAL=3

# Output directories
APG_OUTPUT="./results/tier1_apg/"
PROBE_OUTPUT="./results/tier1_probe_ablation/"

# Codebase paths and their IDs (for action log lookup)
declare -A CODEBASE_IDS=(
    [42]="gen-pipeline-medium-f3784e0c"
    [123]="gen-pipeline-medium-c2777a58"
    [999]="gen-pipeline-medium-fd3fe909"
)

SEEDS=(42 123 999)

# Log file
LOGFILE="./results/tier1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ./results/

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"
}

run_eval() {
    local description="$1"
    shift
    log "START: $description"
    if $PYTHON -m evaluation.run_eval evaluate "$@" 2>&1 | tee -a "$LOGFILE"; then
        log "DONE:  $description"
    else
        log "FAIL:  $description (exit code $?)"
    fi
    echo "" >> "$LOGFILE"
}

# ── Tier 1A: Active-Passive Gap ────────────────────────────────────

run_apg() {
    log "═══════════════════════════════════════════"
    log "TIER 1A: Active-Passive Gap (APG)"
    log "═══════════════════════════════════════════"
    mkdir -p "$APG_OUTPUT"

    # 1A.1: passive-full (3 runs — cheapest, 1 API call each)
    log "--- passive-full (3 runs) ---"
    for seed in "${SEEDS[@]}"; do
        run_eval "passive-full seed=$seed" \
            --model "$MODEL" \
            --codebase "./data/codebase_$seed" \
            --mode passive-full \
            --output "$APG_OUTPUT"
    done

    # 1A.2: passive-oracle (3 runs — ~7 API calls each)
    log "--- passive-oracle (3 runs) ---"
    for seed in "${SEEDS[@]}"; do
        run_eval "passive-oracle seed=$seed" \
            --model "$MODEL" \
            --codebase "./data/codebase_$seed" \
            --mode passive-oracle \
            --budget "$BUDGET" \
            --probe-interval "$PROBE_INTERVAL" \
            --output "$APG_OUTPUT"
    done

    # 1A.3: passive-replay (3 runs — replay active run's action log)
    log "--- passive-replay (3 runs) ---"
    for seed in "${SEEDS[@]}"; do
        cid="${CODEBASE_IDS[$seed]}"
        action_log="./results/openai_gpt-5.3-codex_${cid}_active_action_log.json"

        if [ ! -f "$action_log" ]; then
            log "ERROR: Action log not found: $action_log"
            log "  (Need active run results for passive-replay)"
            continue
        fi

        run_eval "passive-replay seed=$seed" \
            --model "$MODEL" \
            --codebase "./data/codebase_$seed" \
            --mode passive-replay \
            --replay-log "$action_log" \
            --probe-interval "$PROBE_INTERVAL" \
            --output "$APG_OUTPUT"
    done

    log "Tier 1A complete. Results in $APG_OUTPUT"
}

# ── Tier 1B: Probe Condition Ablation ──────────────────────────────

run_probe_ablation() {
    log "═══════════════════════════════════════════"
    log "TIER 1B: Probe Condition Ablation"
    log "═══════════════════════════════════════════"
    mkdir -p "$PROBE_OUTPUT"

    # 1B.1: no_probe (3 runs — final map only, no intermediate probes)
    log "--- no_probe (3 runs) ---"
    for seed in "${SEEDS[@]}"; do
        run_eval "no_probe seed=$seed" \
            --model "$MODEL" \
            --codebase "./data/codebase_$seed" \
            --mode active \
            --track no_probe \
            --budget "$BUDGET" \
            --probe-interval "$PROBE_INTERVAL" \
            --output "$PROBE_OUTPUT"
    done

    # 1B.2: probe_only (3 runs — JSON collected but stripped from context)
    log "--- probe_only (3 runs) ---"
    for seed in "${SEEDS[@]}"; do
        run_eval "probe_only seed=$seed" \
            --model "$MODEL" \
            --codebase "./data/codebase_$seed" \
            --mode active \
            --track probe_only \
            --budget "$BUDGET" \
            --probe-interval "$PROBE_INTERVAL" \
            --output "$PROBE_OUTPUT"
    done

    log "Tier 1B complete. Results in $PROBE_OUTPUT"
}

# ── Main ───────────────────────────────────────────────────────────

main() {
    log "═══════════════════════════════════════════"
    log "ToCS Tier 1 Experiments"
    log "Model: $MODEL"
    log "Budget: $BUDGET, Probe interval: $PROBE_INTERVAL"
    log "═══════════════════════════════════════════"

    # Check API key
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        log "ERROR: OPENAI_API_KEY not set"
        exit 1
    fi

    # Check codebases exist
    for seed in "${SEEDS[@]}"; do
        if [ ! -f "./data/codebase_$seed/ground_truth.json" ]; then
            log "ERROR: Codebase not found: ./data/codebase_$seed"
            log "  Generate with: $PYTHON -m generator --pattern pipeline --complexity medium --seed $seed --output ./data/codebase_$seed"
            exit 1
        fi
    done

    # Check active action logs exist (needed for passive-replay)
    for seed in "${SEEDS[@]}"; do
        cid="${CODEBASE_IDS[$seed]}"
        action_log="./results/openai_gpt-5.3-codex_${cid}_active_action_log.json"
        if [ ! -f "$action_log" ]; then
            log "WARNING: Action log not found: $action_log"
            log "  passive-replay for seed=$seed will be skipped"
        fi
    done

    local target="${1:-all}"
    case "$target" in
        1a|apg)
            run_apg
            ;;
        1b|probe)
            run_probe_ablation
            ;;
        all)
            run_apg
            run_probe_ablation
            ;;
        *)
            echo "Usage: $0 [all|1a|1b]"
            exit 1
            ;;
    esac

    log ""
    log "═══════════════════════════════════════════"
    log "ALL TIER 1 EXPERIMENTS COMPLETE"
    log "Log saved to: $LOGFILE"
    log "═══════════════════════════════════════════"
}

main "$@"
