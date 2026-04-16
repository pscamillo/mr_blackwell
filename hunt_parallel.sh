#!/bin/bash
# =============================================================================
# hunt_parallel.sh - Parallel sieve+GPU pipeline for prime gap search
#
# Author: Camillo / pscamillo
#
# Innovation: runs CPU sieve for range N+1 in background while GPU tests
# range N. Since sieve is CPU-only and PRP test is GPU-only, they don't
# compete for resources. This nearly doubles throughput.
#
# Sequential: sieve(33s) -> GPU(52s) -> sieve(33s) -> GPU(52s) = 85s/range
# Parallel:   sieve(33s) -> [GPU(52s) + sieve(33s)] -> ... = 52s/range
#
# Usage: ./hunt_parallel.sh <start_m> <end_m> <logfile>
# Example: ./hunt_parallel.sh 61700001 100000001 hunt_log3.txt
# =============================================================================

export LC_NUMERIC=C

# Parameters
P=907
D=2190
MINC=100000
SIEVE=18500
MAX_PRIME=200
MIN_MERIT=20

# Command line args
START_M=${1:-61700001}
END_M=${2:-100000001}
LOGFILE=${3:-hunt_parallel.txt}

# Generate filename for a given mstart
make_fname() {
    echo "${P}_${D}_${1}_${MINC}_s${SIEVE}_l200M.txt"
}

# Run sieve for a given mstart
run_sieve() {
    local mstart=$1
    ./combined_sieve -p $P -d $D --mstart $mstart --minc $MINC \
        --max-prime $MAX_PRIME --sieve-length $SIEVE \
        --save-unknowns -qqq
}

# Run GPU test for a given mstart
run_gpu() {
    local mstart=$1
    local fname=$(make_fname $mstart)
    ./gap_test_gpu --unknown-filename "$fname" --min-merit $MIN_MERIT
}

echo "=== Parallel Pipeline: P=$P, sieve=$SIEVE ===" | tee -a "$LOGFILE"
echo "=== Range: $START_M to $END_M ===" | tee -a "$LOGFILE"
echo "=== Started: $(date) ===" | tee -a "$LOGFILE"

# Generate list of mstarts
MSTARTS=()
for m in $(seq $START_M $MINC $END_M); do
    MSTARTS+=($m)
done
TOTAL=${#MSTARTS[@]}
echo "=== Total ranges: $TOTAL ===" | tee -a "$LOGFILE"

if [ $TOTAL -eq 0 ]; then
    echo "No ranges to process."
    exit 0
fi

# =============================================
# Phase 1: Sieve first range (no overlap yet)
# =============================================
CURRENT_IDX=0
CURRENT_M=${MSTARTS[$CURRENT_IDX]}
echo "=== mstart=$CURRENT_M  $(date) ===" >> "$LOGFILE"
echo "[Pipeline] Sieving first range: $CURRENT_M"
run_sieve $CURRENT_M 2>> "$LOGFILE"

# =============================================
# Phase 2: Main pipeline loop
# For each range: GPU test current + sieve next (in parallel)
# =============================================
COMPLETED=0
PIPELINE_START=$(date +%s)

while [ $CURRENT_IDX -lt $TOTAL ]; do
    CURRENT_M=${MSTARTS[$CURRENT_IDX]}
    NEXT_IDX=$((CURRENT_IDX + 1))
    
    if [ $NEXT_IDX -lt $TOTAL ]; then
        # There's a next range: sieve it in background while GPU runs
        NEXT_M=${MSTARTS[$NEXT_IDX]}
        echo "=== mstart=$NEXT_M  $(date) ===" >> "$LOGFILE"
        run_sieve $NEXT_M 2>> "$LOGFILE" &
        SIEVE_PID=$!
        
        # GPU test current range (foreground)
        run_gpu $CURRENT_M >> "$LOGFILE" 2>&1
        
        # Wait for background sieve to finish (should already be done)
        wait $SIEVE_PID
    else
        # Last range: just GPU test, no next sieve needed
        run_gpu $CURRENT_M >> "$LOGFILE" 2>&1
    fi
    
    COMPLETED=$((COMPLETED + 1))
    CURRENT_IDX=$((CURRENT_IDX + 1))
    
    # Progress report every 10 ranges
    if [ $((COMPLETED % 10)) -eq 0 ]; then
        NOW=$(date +%s)
        ELAPSED=$((NOW - PIPELINE_START))
        RATE=$(echo "scale=1; $ELAPSED / $COMPLETED" | bc)
        REMAINING=$(echo "scale=0; ($TOTAL - $COMPLETED) * $ELAPSED / $COMPLETED" | bc)
        REM_MIN=$(echo "scale=1; $REMAINING / 60" | bc)
        echo "[Pipeline] Progress: $COMPLETED/$TOTAL ranges, ${RATE}s/range, ~${REM_MIN} min remaining" | tee -a "$LOGFILE"
    fi
done

PIPELINE_END=$(date +%s)
TOTAL_SEC=$((PIPELINE_END - PIPELINE_START))
TOTAL_MIN=$(echo "scale=1; $TOTAL_SEC / 60" | bc)
AVG=$(echo "scale=1; $TOTAL_SEC / $TOTAL" | bc)

echo "" | tee -a "$LOGFILE"
echo "=== Pipeline Complete ===" | tee -a "$LOGFILE"
echo "=== Ranges: $TOTAL, Time: ${TOTAL_MIN} min, Avg: ${AVG}s/range ===" | tee -a "$LOGFILE"
echo "=== Finished: $(date) ===" | tee -a "$LOGFILE"

# Show best results
echo "" | tee -a "$LOGFILE"
echo "=== Top 10 gaps found ===" | tee -a "$LOGFILE"
grep "^[0-9]" "$LOGFILE" | grep "907#" | sort -k2 -rn | head -10 | tee -a "$LOGFILE"
