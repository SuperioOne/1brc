# 1 Billion Rows Challange - Rust

## x86-64 and AVX2 only, Abomination Edition (0.95107 seconds)

SIMD instructions, libc mmap, CityHash and Rust standard library.

RUSTFLAGS: "-O -Ctarget-cpu=native"

**Test Environment:**
- Ryzen 9 5950x CPU (16C/32T)(x86-64-v3)
- 64GB Memory
- Evo 970 SSD (3400 MB/s Seq. Read)
- Rust 1.80

Best case scenario - measurements.txt is already cached on memory.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt' (20 runs):

        31,989,081      cache-misses                                                            ( +-  0.35% )  (83.34%)
    94,279,303,878      cycles                           #    4.380 GHz                         ( +-  0.02% )  (83.32%)
   189,874,973,170      instructions                     #    2.01  insn per cycle              ( +-  0.00% )  (83.31%)
       515,780,659      branch-misses                                                           ( +-  0.07% )  (83.33%)
    40,944,074,665      all_data_cache_accesses          #    1.902 G/sec                       ( +-  0.01% )  (83.35%)
         21,524.50 msec cpu-clock                        #   22.632 CPUs utilized               ( +-  0.06% )
       515,328,339      branch-misses                                                           ( +-  0.07% )  (83.53%)
               148      context-switches                 #    6.876 /sec                        ( +-  2.93% )

           # Table of individual measurements:
           0.95327 (+0.00220) #
           0.95674 (+0.00568) #
           0.95285 (+0.00178) #
           0.95213 (+0.00106) #
           0.93462 (-0.01645) #
           0.95855 (+0.00748) #
           0.94758 (-0.00349) #
           0.95636 (+0.00529) #
           0.95313 (+0.00207) #
           0.93784 (-0.01323) #
           0.95665 (+0.00558) #
           0.95529 (+0.00422) #
           0.95360 (+0.00253) #
           0.95335 (+0.00228) #
           0.95819 (+0.00712) #
           0.95336 (+0.00230) #
           0.93641 (-0.01466) #
           0.93737 (-0.01370) #
           0.95877 (+0.00770) #
           0.95530 (+0.00424) #

           # Final result:
           0.95107 +- 0.00176 seconds time elapsed  ( +-  0.19% )
```

Worst case scenario - Cold cache, nothing cached on memory. This test is always bottlenecked by storage read speed.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt':

       282,683,548      cache-misses                                                            (83.55%)
    80,481,908,288      cycles                           #    4.377 GHz                         (83.59%)
   203,282,338,210      instructions                     #    2.53  insn per cycle              (82.26%)
       807,543,729      branch-misses                                                           (83.02%)
    48,256,836,556      all_data_cache_accesses          #    2.624 G/sec                       (83.88%)
         18,389.08 msec cpu-clock                        #    4.522 CPUs utilized
       807,284,133      branch-misses                                                           (83.72%)
            99,257      context-switches                 #    5.398 K/sec

       4.066570682 seconds time elapsed

      13.439515000 seconds user
       4.550043000 seconds sys
```

## Testing on your machine

1. Get a x86-64 CPU with AVX2 support. This version does not support ARM.
2. Generate `measurements.txt`. See [https://github.com/gunnarmorling/1brc](https://github.com/gunnarmorling/1brc)
3. Set `RUSTFLAGS` ENV variable to `-Ctarget-cpu=native`. Linux example; `export RUSTFLAGS="-Ctarget-cpu=native"`
4. Run cargo release build by `cargo build --release`.
5. Test it. `./target/release/challange_1brc <PATH_TO_MEASUREMENTS_FILE>`

If you want to use Linux only `./perf_measure.sh` script to generate stats:

6. Audit the script, do not run random scripts from internet. It's only 15 lines.
7. Make sure Linux `perf` tool is installed.
8. Run the script `sudo ./perf_measure.sh <PATH_TO_MEASUREMENTS_FILE>` 

> Dropping memory caches and performance counters requires `sudo` privilages.

