#!/bin/bash

# Usage: ./monitor_disk_peak.sh benchmark_id
MONITOR_DIR="data/$1"
CMD="python $1.py"
shift
CMD="$CMD $@"

if [ -z "$MONITOR_DIR" ] || [ -z "$CMD" ]; then
    echo "Usage: $0 /path/to/directory \"command to run\""
    exit 1
fi

PEAK_FILE=$(mktemp)
echo 0 > "$PEAK_FILE"

monitor_usage() {
    while ps -p "$1" > /dev/null 2>&1; do
        current_size=$(du -sk "$MONITOR_DIR" 2>/dev/null | cut -f1)
        current_size=$((current_size * 1024))  # convert KB to bytes
        peak_size=$(cat "$PEAK_FILE")
        if [ "$current_size" -gt "$peak_size" ]; then
            echo "$current_size" > "$PEAK_FILE"
        fi
        sleep 0.1
    done
}

# Start command in background
bash -c "$CMD" &
CMD_PID=$!

# Start monitoring in background
monitor_usage "$CMD_PID" &
MONITOR_PID=$!

# Wait for command
wait "$CMD_PID"
kill "$MONITOR_PID" 2>/dev/null

# Report result
PEAK_BYTES=$(cat "$PEAK_FILE")
PEAK_MB=$(echo "scale=2; $PEAK_BYTES/1024/1024" | bc)
echo "Peak storage usage in $MONITOR_DIR: $PEAK_MB MB"

rm -f "$PEAK_FILE"
