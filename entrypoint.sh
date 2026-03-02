#!/bin/bash
# Ensure cache and temp dirs exist on the mounted volume to avoid filling root disk
set -e
mkdir -p /runpod-volume/huggingface/hub /runpod-volume/huggingface/xet /runpod-volume/torch /runpod-volume/tmp
exec "$@"
