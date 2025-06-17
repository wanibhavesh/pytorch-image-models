#!/bin/bash

DONEFILE="/tmp/install_done_${SLURM_JOB_ID}"

if [[ $SLURM_LOCALID == 0 ]]; then
  echo "🔧 [0] Installing pip packages inside container..."

  python3 -m pip install --upgrade pip
  pip install -r requirements.txt

  touch "${DONEFILE}"
else
  echo "⏳ [${SLURM_LOCALID}] Waiting for setup to finish..."
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# Execute training script passed to this wrapper
exec "$@"
