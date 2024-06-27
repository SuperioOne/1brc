#!/bin/bash
set -e

if [[ -n "${1}" ]]; then
        EVENTS="cache-misses,cycles,instructions,branch-misses,all_data_cache_accesses,cpu-clock,branch-misses,context-switches";

        echo "Pre-warm for memory cache";
        for i in {0..3}; do 
                ./target/release/challange_1brc "${1}" > /dev/null
        done;

        perf stat -e "${EVENTS}" ./target/release/challange_1brc "${1}" > /dev/null

        echo "Cold Cache Test";
        echo "Dropping memory caches";
        sync;
        echo 1 > "/proc/sys/vm/drop_caches"
        sleep 10; # Give it a time to drop caches

        perf stat -e "${EVENTS}" ./target/release/challange_1brc "${1}" > /dev/null

else
        echo "measurements.txt path not defined";
        echo "See https://github.com/gunnarmorling/1brc to see how to generate measurements.txt";
fi

