#!/bin/sh
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )
echo "Running script in ${SCRIPT_DIR}"

docker run \
    --volume="${SCRIPT_DIR}":/workspace\
    ghcr.io/oxoaresearch/kheops-compiled:latest \
        /kheopscli/bin/kheopscli\
            --project-dir /workspace/src \
            --config-path /workspace/.kheops/input-configuration.json \
            --input-parser julia \
            --output-type diagram \
            --diagram-backend mermaid \
            --diagram-type file \
            --log-level debug
