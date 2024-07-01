# 1 Billion Rows Challange - Rust

## x86-64 and AVX2 only, Abomination Edition (1.37268 seconds)

SIMD instructions, libc mmap, CityHash and Rust standard library.

RUSTFLAGS: "-O -Ctarget-cpu=native"

**Test Hardware:**
- Ryzen 9 5950x CPU (16C/32T)(x86-64-v3)
- 64GB Memory
- Evo 970 SSD (3400 MB/s Seq. Read)

Best case scenario - measurements.txt is already cached on memory.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt' (20 runs):

        38,525,225      cache-misses                                                            ( +-  0.54% )  (83.21%)
   155,401,906,745      cycles                           #    4.437 GHz                         ( +-  0.03% )  (83.27%)
   357,175,281,789      instructions                     #    2.30  insn per cycle              ( +-  0.01% )  (83.41%)
       553,642,583      branch-misses                                                           ( +-  0.16% )  (83.46%)
    79,422,777,469      all_data_cache_accesses          #    2.268 G/sec                       ( +-  0.01% )  (83.48%)
         35,021.39 msec cpu-clock                        #   25.513 CPUs utilized               ( +-  0.09% )
       553,427,669      branch-misses                                                           ( +-  0.17% )  (83.28%)
               193      context-switches                 #    5.511 /sec                        ( +-  3.34% )

           # Table of individual measurements:
           1.36296 (-0.00971) #
           1.38376 (+0.01108) #
           1.36718 (-0.00550) #
           1.36102 (-0.01166) #
           1.36893 (-0.00375) #
           1.36973 (-0.00295) #
           1.36941 (-0.00327) #
           1.37097 (-0.00171) #
           1.37227 (-0.00041) #
           1.37732 (+0.00464) #
           1.37311 (+0.00043) #
           1.36608 (-0.00660) #
           1.38409 (+0.01141) #
           1.37324 (+0.00056) #
           1.37255 (-0.00013) #
           1.37465 (+0.00197) #
           1.37622 (+0.00354) #
           1.39284 (+0.02016) #
           1.36779 (-0.00489) #
           1.36948 (-0.00320) #

           # Final result:
           1.37268 +- 0.00168 seconds time elapsed  ( +-  0.12% )
```

Worst case scenario - Cold cache, nothing cached on memory. This test is always bottlenecked by storage read speed.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt':

       293,594,283      cache-misses                                                            (83.36%)
   123,866,390,722      cycles                           #    4.467 GHz                         (83.38%)
   371,918,751,853      instructions                     #    3.00  insn per cycle              (83.11%)
       875,992,043      branch-misses                                                           (83.16%)
    87,988,682,818      all_data_cache_accesses          #    3.173 G/sec                       (83.60%)
         27,730.85 msec cpu-clock                        #    6.790 CPUs utilized
       875,586,936      branch-misses                                                           (83.42%)
           100,816      context-switches                 #    3.636 K/sec

       4.083973242 seconds time elapsed

      22.250537000 seconds user
       5.019990000 seconds sys
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

