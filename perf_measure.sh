#!/bin/bash
set -e

if [[ -n "${1}" ]]; then
    EVENTS="cache-misses,cycles,instructions,branch-misses,all_data_cache_accesses,cpu-clock,branch-misses,context-switches";

    echo "Pre-warm";
    for i in {1..3}; do
        echo "Pre-Iter...${i}";
        ./target/release/challenge_1brc "${1}" > /dev/null
    done;

    echo "Benchmarking";
    perf stat -e "${EVENTS}" -r 20 --table ./target/release/challenge_1brc "${1}" > /dev/null

    echo "Cold Cache Test";
    echo "Dropping memory caches";
    sync;
    echo 1 > "/proc/sys/vm/drop_caches"
    sleep 5;

    perf stat -e "${EVENTS}" ./target/release/challenge_1brc "${1}" > /dev/null
else
    echo "measurements.txt path not defined";
    echo "See https://github.com/gunnarmorling/1brc to see how to generate measurements.txt";
fi

