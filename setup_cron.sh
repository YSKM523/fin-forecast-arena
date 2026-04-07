#!/usr/bin/env bash
# Set up a cron job to run the daily pipeline at 22:00 UTC (5 PM EST)
# on weekdays (Mon–Fri).
#
# Usage:  bash setup_cron.sh          # install the cron entry
#         bash setup_cron.sh remove    # remove the cron entry

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO_DIR/venv/bin/activate"
SCRIPT="$REPO_DIR/pipeline/daily_run.py"
LOG="$REPO_DIR/results/logs/daily_run.log"
MARKER="# fin-forecast-arena daily pipeline"

mkdir -p "$REPO_DIR/results/logs"

CRON_LINE="0 22 * * 1-5  . $VENV && python $SCRIPT >> $LOG 2>&1  $MARKER"

if [[ "${1:-}" == "remove" ]]; then
    crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
    echo "Cron entry removed."
    exit 0
fi

# Add entry if not already present
( crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$CRON_LINE" ) | crontab -

echo "Cron entry installed:"
echo "  $CRON_LINE"
echo ""
echo "Schedule: Mon–Fri at 22:00 UTC (17:00 EST)"
echo "Log file: $LOG"
echo ""
echo "Verify with:  crontab -l"
echo "Remove with:  bash $0 remove"
