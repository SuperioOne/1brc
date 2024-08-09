# 1 Billion Rows Challange - Rust

## What's new

I find the AVX2 line iterator pretty useful, and I decided to move it to my own [algorithms repo](https://github.com/SuperioOne/algorithms) for future uses.
Also, it now has a fallback algorithm based on SWAR for other CPU architectures and older x86-64 CPUs.

For previous inlined AVX2 only version, see [`avx2_only` tag](https://github.com/SuperioOne/1brc/tree/avx2_only)

## x86-64-v3 with AVX2 (0.95855 seconds)

SIMD instructions, libc mmap, CityHash and Rust standard library.

RUSTFLAGS: "-O -Ctarget-cpu=x86-64-v3"

**Test Environment:**
- Ryzen 9 5950x CPU (16C/32T)(x86-64-v3)
- 64GB Memory
- Evo 970 SSD (3400 MB/s Seq. Read)
- Rust 1.80.1

Best case scenario - measurements.txt is already cached on memory.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt' (20 runs):

        36,300,329      cache-misses                                                            ( +-  0.53% )  (83.23%)
    97,810,561,440      cycles                           #    4.452 GHz                         ( +-  0.05% )  (83.25%)
   190,712,123,166      instructions                     #    1.95  insn per cycle              ( +-  0.01% )  (83.29%)
       542,035,366      branch-misses                                                           ( +-  0.04% )  (83.39%)
    37,941,771,530      all_data_cache_accesses          #    1.727 G/sec                       ( +-  0.01% )  (83.53%)
         21,969.19 msec cpu-clock                        #   22.919 CPUs utilized               ( +-  0.09% )
       542,153,328      branch-misses                                                           ( +-  0.05% )  (83.51%)
               144      context-switches                 #    6.555 /sec                        ( +-  3.24% )

           # Table of individual measurements:
           0.94001 (-0.01854) #
           0.95140 (-0.00715) #
           0.94972 (-0.00883) #
           0.94955 (-0.00900) #
           0.95507 (-0.00348) #
           0.94923 (-0.00932) #
           0.95880 (+0.00025) #
           0.96161 (+0.00306) #
           0.96030 (+0.00175) #
           0.95484 (-0.00370) #
           0.95875 (+0.00021) #
           0.96139 (+0.00284) #
           0.99126 (+0.03271) #
           0.96414 (+0.00559) #
           0.95875 (+0.00021) #
           0.95207 (-0.00648) #
           0.95972 (+0.00117) #
           0.96678 (+0.00823) #
           0.96704 (+0.00849) #
           0.96054 (+0.00199) #

           # Final result:
           0.95855 +- 0.00228 seconds time elapsed  ( +-  0.24% )

```

Worst case scenario - Cold cache, nothing cached on memory.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt':

       277,784,366      cache-misses                                                            (82.66%)
    83,212,123,901      cycles                           #    4.430 GHz                         (83.42%)
   204,322,015,153      instructions                     #    2.46  insn per cycle              (83.69%)
       826,555,011      branch-misses                                                           (83.75%)
    44,637,535,477      all_data_cache_accesses          #    2.376 G/sec                       (82.61%)
         18,783.41 msec cpu-clock                        #    4.619 CPUs utilized
       827,751,960      branch-misses                                                           (83.90%)
           100,820      context-switches                 #    5.368 K/sec

       4.066922345 seconds time elapsed

      14.087471000 seconds user
       4.297025000 seconds sys
```

## x86-64 without any SIMD tricks (1.07852)

libc mmap, CityHash and Rust standard library.

RUSTFLAGS: "-O -Ctarget-cpu=x86-64"

**Test Environment:**
- Ryzen 9 5950x CPU (16C/32T)
- 64GB Memory
- Evo 970 SSD (3400 MB/s Seq. Read)
- Rust 1.80.1

Best case scenario - measurements.txt is already cached on memory.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt' (20 runs):

        31,335,865      cache-misses                                                            ( +-  0.44% )  (83.26%)
   114,884,685,620      cycles                           #    4.446 GHz                         ( +-  0.06% )  (83.20%)
   230,630,440,560      instructions                     #    2.01  insn per cycle              ( +-  0.01% )  (83.28%)
       783,966,172      branch-misses                                                           ( +-  0.01% )  (83.40%)
    44,204,291,969      all_data_cache_accesses          #    1.711 G/sec                       ( +-  0.02% )  (83.48%)
         25,840.46 msec cpu-clock                        #   23.959 CPUs utilized               ( +-  0.09% )
       784,057,687      branch-misses                                                           ( +-  0.01% )  (83.56%)
               160      context-switches                 #    6.192 /sec                        ( +-  3.40% )

           # Table of individual measurements:
           1.06180 (-0.01671) #
           1.08275 (+0.00423) #
           1.06213 (-0.01638) #
           1.07163 (-0.00689) #
           1.07277 (-0.00575) #
           1.06830 (-0.01022) #
           1.06949 (-0.00902) #
           1.07045 (-0.00807) #
           1.06786 (-0.01066) #
           1.07656 (-0.00195) #
           1.07785 (-0.00066) #
           1.07309 (-0.00543) #
           1.07582 (-0.00269) #
           1.07799 (-0.00052) #
           1.07297 (-0.00555) #
           1.11058 (+0.03207) #
           1.09281 (+0.01430) #
           1.12555 (+0.04703) #
           1.07851 (-0.00000) #
           1.08138 (+0.00287) #

           # Final result:
           1.07852 +- 0.00345 seconds time elapsed  ( +-  0.32% )
```

Worst case scenario - Cold cache, nothing cached on memory.
```
 Performance counter stats for './target/release/challange_1brc ./measurements.txt':

       274,598,290      cache-misses                                                            (82.63%)
    98,389,092,251      cycles                           #    4.438 GHz                         (82.68%)
   244,232,067,759      instructions                     #    2.48  insn per cycle              (83.54%)
     1,074,358,436      branch-misses                                                           (84.12%)
    53,795,941,392      all_data_cache_accesses          #    2.426 G/sec                       (83.44%)
         22,171.22 msec cpu-clock                        #    5.438 CPUs utilized
     1,074,669,965      branch-misses                                                           (83.62%)
            99,908      context-switches                 #    4.506 K/sec

       4.076780812 seconds time elapsed

      17.503729000 seconds user
       4.241538000 seconds sys
```

## Testing on your machine

1. Generate `measurements.txt`. See [https://github.com/gunnarmorling/1brc](https://github.com/gunnarmorling/1brc)
2. Set `RUSTFLAGS` ENV variable to `-Ctarget-cpu=native`. Linux example; `export RUSTFLAGS="-Ctarget-cpu=native"`
3. Run cargo release build by `cargo build --release`.
4. Test it. `./target/release/challange_1brc <PATH_TO_MEASUREMENTS_FILE>`

If you want to use Linux only `./perf_measure.sh` script to generate stats:

5. Audit the script, do not run random scripts from internet. It's only 15 lines.
6. Make sure Linux `perf` tool is installed.
7. Run the script `sudo ./perf_measure.sh <PATH_TO_MEASUREMENTS_FILE>` 

> Dropping memory caches and performance counters requires `sudo` privilages.

