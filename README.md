# 1 Billion Rows Challange - Rust

## Ferris, Take the Compiler Edition

No fancy manual SIMD instructions and bitwise tricks (yet). Only libc mmap, CityHash and Rust standard library.

RUSTFLAGS: "-O -Ctarget-cpu=native"

**Test Hardware:**
- Ryzen 9 5950x CPU (16C/32T)(x86-64-v3)
- 64GB Memory
- Evo 970 SSD (3400 MB/s Seq. Read)

Best case scenario - measurements.txt already cached to memory.
```
        42,205,426      cache-misses                                                            (83.26%)
   211,780,686,177      cycles                           #    4.405 GHz                         (83.30%)
   437,100,686,745      instructions                     #    2.06  insn per cycle              (83.31%)
     1,581,342,514      branch-misses                                                           (83.40%)
    88,892,189,359      all_data_cache_accesses          #    1.849 G/sec                       (83.38%)
         48,072.68 msec cpu-clock                        #   26.723 CPUs utilized
     1,581,670,473      branch-misses                                                           (83.42%)
             2,781      context-switches                 #   57.850 /sec

       1.798937484 seconds time elapsed
```

Worst case scenario - Cold memory, no cache (bottlenecked by SSD)
```
       314,793,007      cache-misses                                                            (83.33%)
   171,254,713,875      cycles                           #    4.461 GHz                         (83.17%)
   452,349,949,477      instructions                     #    2.64  insn per cycle              (83.37%)
     1,966,005,731      branch-misses                                                           (83.31%)
   104,548,157,509      all_data_cache_accesses          #    2.723 G/sec                       (83.85%)
         38,387.85 msec cpu-clock                        #    9.321 CPUs utilized
     1,965,995,329      branch-misses                                                           (83.01%)
           102,719      context-switches                 #    2.676 K/sec

       4.118434308 seconds time elapsed
```

