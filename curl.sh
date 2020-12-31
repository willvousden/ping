#!/usr/bin/env bash

while sleep 1; do 
    curl \
        -w @curl-format.txt \
        -o /dev/null \
        -m 4 \
        -s \
        "$1" \
        | ts '%Y-%m-%dT%H:%M:%.S'
done
