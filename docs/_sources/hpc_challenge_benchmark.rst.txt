==============================================================================
HPC Challenge Benchmark
==============================================================================

1 - Linpack
==============================================================================

The Linpack benchmark is a measure of computer floating point execution efficiency and is the basis for the Top 500 supercomputer rankings.

1.1 - Usage
-----------------------
Example Repo : Linpack_

.. _Linpack: https://netlib.org/benchmark/hpl/index.html

Example Script::

    cat <<EOF > batchScript.sh
    #!/bin/bash
    #SBATCH --partition=arm-kunpeng920
    #SBATCH --time=00:25:00
    #SBATCH --ntasks=128
    #SBATCH --nodes=1
    module load GCC/12.1.0 OpenBLAS/0.3.21 OpenMPI/4.1.3
    mpirun --allow-run-as-root -npernode 8 -x OMP_NUM_THREADS=16 ./xhpl
    EOF
    sbatch batchScript.sh

1.2 - Results
-----------------------
.. list-table:: LINPACK-Test
   :widths: 25 25 25 55 25
   :header-rows: 1

   * - CPU
     - Compiler Combination
     - Number of Nodes
     - Number of Cores
     - Test Result
   * - arm-kunpeng920
     - GCC/12.1.0
     - 1
     - 128 (16 processes, 8 threads per process)
     - 3.0346e+02Gflops
   * - arm-kunpeng920
     - GCC/12.1.0
     - 1
     - 128 (128 processes, 1 threads per process)
     - 3.9471e+01Gflops
   * - arm-kunpeng920
     - GCC/12.1.0
     - 1
     - 128 (8 processes, 16 threads per process)
     - 7.2177e+02Gflops
   * - arm-kunpeng920
     - GCC/12.1.0
     - 1
     - 128 (4 processes, 32 threads per process)
     - 6.0959e+02Gflops
   * - arm-kunpeng920
     - GCC/12.1.0
     - 1
     - 128 (64 processes, 2 threads per process)
     - 7.8394e+01Gflops


2 - STREAM: Sustainable Memory Bandwidth in High Performance Computers
==============================================================================

2.1 - Usage
-----------------------------------
Example Repo : STREAM_

.. _STREAM: https://github.com/jeffhammond/STREAM

Test Script::

   cat <<EOF > batchscript.sh
   #!/bin/bash
   #SBATCH --partition=arm-kunpeng920
   #SBATCH --time=00:10:00
   #SBATCH --ntasks=128
   #SBATCH --nodes=1
   module load GCC/12.1.0 OpenMPI
   gcc -fopenmp -O3 -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 -mcmodel=large stream.c -o stream_c
   ./stream_c
   EOF
   sbatch batchscript.sh

2.2 - Results
-----------------------------------

Output::

   *Not optimised for performance*
   -------------------------------------------------------------
   STREAM version $Revision: 5.10 $
   -------------------------------------------------------------
   This system uses 8 bytes per array element.
   -------------------------------------------------------------
   Array size = 80000000 (elements), Offset = 0 (elements)
   Memory per array = 610.4 MiB (= 0.6 GiB).
   Total memory required = 1831.1 MiB (= 1.8 GiB).
   Each kernel will be executed 20 times.
   The *best* time for each kernel (excluding the first iteration)
   will be used to compute the reported bandwidth.
   -------------------------------------------------------------
   Number of Threads requested = 128
   Number of Threads counted = 128
   -------------------------------------------------------------
   Your clock granularity/precision appears to be 1 microseconds.
   Each test below will take on the order of 10418 microseconds.
      (= 10418 clock ticks)
   Increase the size of the arrays if this shows that
   you are not getting at least 20 clock ticks per test.
   -------------------------------------------------------------
   WARNING -- The above is only a rough guideline.
   For best results, please be sure you know the
   precision of your system timer.
   -------------------------------------------------------------
   Function    Best Rate MB/s  Avg time     Min time     Max time
   Copy:          141140.7     0.011449     0.009069     0.013668
   Scale:         150675.2     0.010752     0.008495     0.013169
   Add:           128894.4     0.016744     0.014896     0.020257
   Triad:         143679.0     0.016380     0.013363     0.023785
   -------------------------------------------------------------
   Solution Validates: avg error less than 1.000000e-13 on all three arrays
   -------------------------------------------------------------


