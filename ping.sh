#!/usr/bin/env bash

ping $1 | ts '[%Y-%m-%dT%H:%M:%.S]'
