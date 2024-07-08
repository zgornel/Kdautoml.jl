#!/bin/sh
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )
KHEOPS_DIR=$1

if [ -z "$1" ]; then
    echo "First argument (Kheops directory) doesn’t exist or it’s empty."
    exit
fi

echo "Running script in ${SCRIPT_DIR} using Kheops@${KHEOPS_DIR}"
julia "${KHEOPS_DIR}/kheopscliw.jl" \
    --project-dir "${SCRIPT_DIR}/src" \
    --config-path ./input-configuration.json \
    --input-parser julia \
    --output-type diagram \
    --diagram-type file \
    --diagram-backend mermaid \
    --log-level debug
