#!/bin/bash

args=""
args+=" --ps-host cnode1"
args+=" --worker-host cnode2"

python example.py --job-name worker --task-index 0 $args
