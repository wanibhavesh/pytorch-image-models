#!/bin/bash

LOG_FILE=$1
INTERVAL=300  # Log every 300 seconds

mkdir -p "$(dirname "$LOG_FILE")"

echo "timestamp,power.draw [W],utilization.gpu [%],temperature.gpu,clocks.current.sm [MHz],gpu.name" > "$LOG_FILE"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

while true; do
    nvidia-smi \
        --query-gpu=timestamp,power.draw,utilization.gpu,temperature.gpu,clocks.sm \
        --format=csv,noheader,nounits \
    | awk -v name="$GPU_NAME" -F, '{ print $1 "," $2 "," $3 "," $4 "," $5 "," name }' >> "$LOG_FILE"
    sleep "$INTERVAL"
done
