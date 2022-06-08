Cluster usage
=============

Software stack overview
-----------------------

HAICGU uses a software stack based on EasyBuild_. The recipes for building software (EasyConfigs) can be found `here`__.

.. _EasyBuild: https://easybuild.io/

.. _juawei-easyconfigs: https://gitlab.jsc.fz-juelich.de/nassyr1/juawei-easyconfigs/

__ juawei-easyconfigs_

The built software is made accessible to users as modules with LMod_. 

.. _LMod: https://www.tacc.utexas.edu/research-development/tacc-projects/lmod

Access to compute nodes is provided using the `SLURM workload manager`__.

.. _SLURM: https://slurm.schedmd.com

__ SLURM_

Using modules
-------------

When on the dev node (``guoehi-dev``), list the available modules by typing::

    module avail

You will see the following output::

    $ module avail

    [...]

    ---------------------------------------------- Compilers in Stage 2021a ----------------------------------------------
       GCC/9.3.0    GCC/10.3.0    GCC/11.1.0 (D)    armlinux/21.1

    -------------------------------------------- Core modules in Stage 2021a ---------------------------------------------
       EasyBuild/4.3.4        Java/8.292.10        armlinux-install/21.1    help2man/1.48.3    zsh/5.8
       EasyBuild/4.5.2 (D)    Java/11.0.10  (D)    flex/2.6.4               tmux/3.2a

    --------------------------------------------------- Architectures ----------------------------------------------------
       Architecture/Kunpeng920 (S)    Architecture/somearch (S,D)

    --------------------------------------------------- Custom modules ---------------------------------------------------
       arm-optimized-routines/21.02 (L)

      Where:
       D:  Default Module
       L:  Module is loaded
       S:  Module is Sticky, requires --force to unload or purge

    [...]


You can load modules with ``module load ModuleName``.
The modules are organized hierarchically - after loading a compiler, more modules will become available::

    $ module load GCC/11.1.0
    $ module avail

    [...]

    --------------------------------------- MPI runtimes available for GCC 11.1.0 ----------------------------------------
       OpenMPI/4.1.2

    ------------------------------------------ Modules compiled with GCC 11.1.0 ------------------------------------------
       Autotools/20210330       Mako/1.1.4                   Python/3.9.4                   giflib/5.2.1                libpfm/4.11.1-f6500e77
       Boost/1.78.0             Mesa/21.0.3                  Qt5/5.15.2                     git/2.31.1                  libvpx/1.10.0
       CFITSIO/3.49             Meson/0.57.1-Python-3.9.4    Rust/1.52.1                    graphene/1.10.6             libwebp/1.2.0
       CMake/3.20.0             NSPR/4.30                    Tcl/8.6.11                     graphite2/1.3.14            libyaml/0.2.5
       Doxygen/1.9.1            NSS/3.63                     UCX/1.11.2                     help2man/1.48.3      (D)    numactl/2.0.14
       Eigen/3.3.9              Ninja/1.10.2                 X11/20210331                   hwloc/2.4.1                 opus/1.3.1-7b05f44f
       GEOS/3.9.1               OpenBLAS/0.3.19              Xerces-C++/3.2.3               libarchive/3.5.1            pkgconf/1.8.0
       GMP/6.2.1                OpenJPEG/2.4.0               cURL/7.75.0                    libffi/3.3                  poppler/22.01.0
       GSL/2.6                  PAPI/6.0.0.1-70887df7        double-conversion/3.1.5        libgit2/1.1.0               re2c/2.1.1
       HDF5/1.12.0-serial       PCRE2/10.36                  elfutils/0.183                 libglvnd/1.3.2              texlive/20210324
       ImageMagick/7.0.11-11    Perl/5.32.1                  flex/2.6.4              (D)    libiconv/1.16               unzip/6.0
       LLVM/12.0.0              PostgreSQL/13.2              fmt/7.1.3                      libmicrohttpd/0.9.72

    [...]

And then after loading an MPI runtime (Currently only OpenMPI), the rest of the modules will become visible::

    [...]

    ---------------------------------- Modules built with GCC 11.1.0 and OpenMPI 4.1.2 -----------------------------------
       Boost/1.75.0-Python-3.9.4        IOR/3.3.0                    ScaLAPACK/2.1.0-OpenBLAS-0.3.19    mpi4py/3.0.3-Python-3.9.4
       FFTW/3.3.9                       OpenCV/4.5.2-Python-3.9.4    SciPy-Stack/2021a-Python-3.9.4     netCDF/4.7.4
       HDF5/1.12.0               (D)    R/4.1.2                      Valgrind/3.17.0

    [...]

AI software stack
-----------------

The AI software stack has been partly integrated into the EasyBuild software stack, it is available with GCC 9.3.0. Load::

    module load GCC/9.3.0 OpenMPI CANN-Toolkit

This will set the necessary environment variables to use the CANN toolkit (AscendCL, ...). This does not load TensorFlow or PyTorch adapters

Using SLURM
-----------

In order to run your application on the actual compute nodes, you will need to submit jobs using SLURM. 

List information about the available partitions and nodes with ``sinfo``::

    $ sinfo
    PARTITION       AVAIL  TIMELIMIT  NODES  STATE NODELIST
    arm-kunpeng920*    up   infinite     28   idle cn[01-28]
    a800-9000          up   infinite      1   idle ml01
    a800-3000          up   infinite      1   idle ml02

As you can see, currently there are 3 partitions available: 

- ``arm-kunpeng920``, currently consisting of 28 standard compute nodes ``cn[01-28]``
- ``a800-9000``, currently consisting of 1 Atlas 800 Training Server (Model: 9000) node ``ml01``
- ``a800-9000``, currently consisting of 1 Atlas 800 Inference Server (Model: 9000) node ``ml02``

You can submit jobs using either the ``srun`` or ``sbatch`` commands.

``srun`` is used to run commands directly::

    $ srun -p arm-kunpeng920 hostname
    cn01.guoehi.cluster

``sbatch`` is used to run batch scripts::

    $ cat <<EOF > batchscript.sh
    > #!/bin/bash
    > #SBATCH --partition=a800-9000
    > #SBATCH --time=00:01:00
    > #SBATCH --ntasks=1
    > #SBATCH --nodes=1
    > npu-smi info
    > EOF
    $ sbatch batchscript.sh 
    Submitted batch job 595
    $ cat slurm-595.out 
    +------------------------------------------------------------------------------------+
    | npu-smi 1.8.21                   Version: 20.2.2.spc001                            |
    +----------------------+---------------+---------------------------------------------+
    | NPU   Name           | Health        | Power(W)   Temp(C)                          |
    | Chip                 | Bus-Id        | AICore(%)  Memory-Usage(MB)  HBM-Usage(MB)  |
    +======================+===============+=============================================+
    | 0     910A           | OK            | 68.6       36                               |
    | 0                    | 0000:C1:00.0  | 0          591  / 14795      0    / 32768   |
    +======================+===============+=============================================+
    | 1     910A           | OK            | 63.7       31                               |
    | 0                    | 0000:81:00.0  | 0          303  / 15177      0    / 32768   |
    +======================+===============+=============================================+
    | 2     910A           | OK            | 66.1       31                               |
    | 0                    | 0000:41:00.0  | 0          1821 / 15177      0    / 32768   |
    +======================+===============+=============================================+
    | 3     910A           | OK            | 65.7       37                               |
    | 0                    | 0000:01:00.0  | 0          3168 / 15088      0    / 32768   |
    +======================+===============+=============================================+
    | 4     910A           | OK            | 66.7       35                               |
    | 0                    | 0000:C2:00.0  | 0          295  / 14795      0    / 32768   |
    +======================+===============+=============================================+
    | 5     910A           | OK            | 63.7       29                               |
    | 0                    | 0000:82:00.0  | 0          455  / 15177      0    / 32768   |
    +======================+===============+=============================================+
    | 6     910A           | OK            | 66.1       29                               |
    | 0                    | 0000:42:00.0  | 0          1517 / 15177      0    / 32768   |
    +======================+===============+=============================================+
    | 7     910A           | OK            | 65.1       36                               |
    | 0                    | 0000:02:00.0  | 0          3319 / 15088      0    / 32768   |
    +======================+===============+=============================================+

You can view the queued jobs by calling ``squeue``::

    $ cat <<EOF > batchscript.sh
    > #!/bin/bash
    > #SBATCH --partition=a800-9000
    > #SBATCH --time=00:01:00
    > #SBATCH --ntasks=1
    > #SBATCH --nodes=1
    > echo waiting
    > sleep 5
    > echo finished waiting
    > EOF
    $ sbatch batchscript.sh 
    Submitted batch job 597
    $ squeue
                 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
                   597 a800-9000 batchscr  snassyr  R       0:01      1 ml01

For more information on how to use SLURM, please read the documentation_

.. _documentation: https://slurm.schedmd.com/documentation.html
