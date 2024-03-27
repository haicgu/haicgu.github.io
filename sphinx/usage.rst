Cluster usage
=============

Software stack overview
-----------------------

HAICGU uses a software stack based on EasyBuild_. The recipes for building software (EasyConfigs) can be found `here`__.

.. _EasyBuild: https://easybuild.io/

.. _sn-easyconfigs: https://github.com/stepannassyr/sn-easyconfigs

__ sn-easyconfigs_

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

    --------------------- Compilers in Stage 2022a ----------------------
       BiSheng-compiler/2.3.0    GCC/9.5.0    GCC/12.1.0 (D)    armlinux/22.0.1

    -------------------- Core modules in Stage 2022a --------------------
       EasyBuild/4.5.5        alplompi/22.0.1            gompi/2022a.12  (D)    tmux/3.3a
       Java/8.292.10          armlinux-install/22.0.1    goolf/2022a.9          zsh/5.8.1
       Java/11.0.15    (D)    flex/2.6.4                 goolf/2022a.12  (D)
       alompi/22.0.1          gompi/2022a.9              help2man/1.49.2

    --------------------------- Architectures ---------------------------
       Architecture/Kunpeng920 (S)    Architecture/somearch (S,D)

    -------------------------- Custom modules ---------------------------
       arm-optimized-routines/21.02 (L)

      Where:
       D:  Default Module
       L:  Module is loaded
       S:  Module is Sticky, requires --force to unload or purge

    [...]


You can load modules with ``module load ModuleName``.
The modules are organized hierarchically - after loading a compiler, more modules will become available::

    $ module load GCC/12.1.0
    $ module avail

    [...]

    --------------- MPI runtimes available for GCC 12.1.0 ---------------
       OpenMPI/4.1.3

    ----------------- Modules compiled with GCC 12.1.0 ------------------
       Autotools/20220509                absl-py/1.0.0-Python-3.10.4
       Bazel/4.2.2                       c-ares/1.18.1
       Bazel/5.1.1                (D)    cURL/7.83.0
       BazelWIT/0.26.1                   dm-tree/0.1.7-Python-3.10.4
       CMake/3.23.1                      double-conversion/3.2.0
       Eigen/3.4.0                       flatbuffers-python/2.0-Python-3.10.4
       GMP/6.2.1                         flatbuffers/2.0.0
       JsonCpp/1.9.5                     flex/2.6.4                           (D)
       Meson/0.62.1-Python-3.10.4        giflib/5.2.1
       Ninja/1.10.2                      git/2.36.1
       OpenBLAS/0.3.20                   help2man/1.49.2                      (D)
       Perl/5.34.1                       hwloc/2.7.1
       Pillow/9.1.1-Python-3.10.4        libffi/3.4.2
       PostgreSQL/14.2                   libyaml/0.2.5
       PyYAML/6.0-Python-3.10.4          lz4/1.9.3
       Python/3.10.4                     nghttp2/1.47.0
       Rust/1.60.0                       nsync/1.24.0
       Tcl/8.6.12                        numactl/2.0.14
       UCX/1.12.1                        protobuf-python/3.20.1-Python-3.10.4
       X11/20220509                      ray-deps/1.12.0-Python-3.10.4
       Zip/3.0                           unzip/6.0
       abseil-cpp/20210324.1

    [...]

And then after loading an MPI runtime (Currently only OpenMPI), the rest of the modules will become visible::

    [...]


    ---------- Modules built with GCC 12.1.0 and OpenMPI 4.1.3 ----------
       Arrow/7.0.0-Python-3.10.4          SciPy-Stack/2022a-Python-3.10.4
       Boost/1.79.0-Python-3.10.4         bokeh/2.4.2-Python-3.10.4
       FFTW/3.3.10                        dask/2022.5.0-Python-3.10.4
       HDF5/1.12.2                        h5py/3.6.0-Python-3.10.4
       ScaLAPACK/2.2.0-OpenBLAS-0.3.20    ray-project/1.12.0-Python-3.10.4

    [...]

.. note::
   There are multiple software stages available (2021a, 2022a), but only the current stage is supported (currently 2022a). You can use load a different stage with
  
   ``. /software/switch_stage.sh -s <stage>``

AI software stack
-----------------

The AI software stack has been partly integrated into the EasyBuild software stack, it is available with GCC 9.5.0. Load::

    module load GCC/9.5.0 OpenMPI CANN-Toolkit

This will set the necessary environment variables to use the CANN toolkit (AscendCL, ...).

You can then load NPU-accelerated AI frameworks.

For TensorFlow 1.15.0 please load::

    module load TensorFlow-CANN/1.15.0

For TensorFlow 2.4.1 please load::

    module load TensorFlow-CANN/2.4.1

For PyTorch 1.5.0 please load::

    module load PyTorch-CANN/1.5.0

.. warning::
   Loading multiple Frameworks or Framework versions at the same time can lead to issues, please make sure to unload one framework with ``module unload <framework module>`` before loading another

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
- ``a800-3000``, currently consisting of 1 Atlas 800 Inference Server (Model: 3000) node ``ml02``

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

Other software
--------------

ArmIE:

To make the module available please use::

    $ module use /software/tools/armie-22.0/modulefiles

You can then load the module with::

    $ module load armie
