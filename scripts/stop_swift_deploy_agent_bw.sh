#!/usr/bin/env bash
# Gracefully stop the ms-swift deploy service launched by launch_swift_deploy_agent_bw.sh.

set -euo pipefail

PORT="${PORT:-8000}"
PATTERN="swift.*deploy.*--port ${PORT}|swift/cli/deploy.py.*--port ${PORT}|launch_swift_deploy_agent_bw.sh"

mapfile -t PIDS < <(pgrep -f "$PATTERN" || true)

if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "No swift deploy service found for port ${PORT}."
    exit 0
fi

echo "Stopping swift deploy service on port ${PORT}: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    if [[ "$pid" != "$$" ]]; then
        kill -TERM "$pid" 2>/dev/null || true
    fi
done

for _ in {1..30}; do
    sleep 1
    if ! pgrep -f "$PATTERN" >/dev/null; then
        echo "Stopped."
        exit 0
    fi
done

echo "Service did not exit after 30 seconds. Remaining processes:"
pgrep -af "$PATTERN" || true
echo "If needed, run with FORCE=1 to send SIGKILL."

if [[ "${FORCE:-0}" == "1" ]]; then
    mapfile -t REMAINING < <(pgrep -f "$PATTERN" || true)
    for pid in "${REMAINING[@]}"; do
        if [[ "$pid" != "$$" ]]; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    echo "Force stopped."
fi
