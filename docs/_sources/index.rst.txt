.. HAICGU documentation master file, created by
   sphinx-quickstart on Wed Jun  8 11:06:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HAICGU cluster documentation
==================================

Huawei AI and Computing at Goethe University (HAICGU) is a cluster based on Huawei HPC solution.
This cluster is installed at the Goethe University of Frankfurt and mainteained by Open Edge and HPC initiative (OEHI).
The Goethe University Frankfurt is part of the National High-Performance Computing Alliance. 

The following listshows the currently installed nodes and servers of the cluster:

#. 28x Standard Compute Node – TaiShan 200 (Model 2280)

   * 2 x Kunpeng 920 processor (ARMv8 AArch64; 64 cores; 2.6GHz; 180W)
   * 28GB main memory (16x 8GB, one DIMM per channel)
   * 1x 100Gbit/s EDR Infiniband HCA
   

#. 1x	Development Compute Node – TaiShan 200 (Model 2280)

   * 2 x Kunpeng 920 processor (ARMv8 AArch64; 64 cores; 2.6GHz; 180W)
   * 128GB main memory (16x 8GB, one DIMM per channel)
   * 1x 100Gbit/s EDR Infiniband HCA
   * 2 x 960GB SSD SATA 6Gb/s

#. 2x	IO Node – TaiShan 200 (Model 5280)

   * Metadata storage: 2x 960GB SSD SATA 6Gb/s (RAID 1)
   * Object storage: 32x 1.2 TB HDD SAS 12Gb/s; 10.000rpm (RAID 10)
   * Mgmt. storage: 4x 1.2 TB HDD SAS 12Gb/s; 10.000rpm (RAID 10)

#. 1x	AI Training Node – Atlas 800 (Model 9000)

   * 4x Kunpeng 920 processor (ARMv8 AArch64)
   * Neural Processing Unit (NPU): 8x Huawei Ascend 910 with 32 AI cores and 32GB HBM2 memory
   * 1024GB main memory: 32x 32GB DDR4 2933MHz RDIMM
   * Local storage: 2x 960 GB SSD SATA 6Gb/s
   * Data storage: 4x 3.2TB SSD NVMe

#. 1x	AI Inference Node – Atlas 800 (Model 3000)
   
   * 2 x Kunpeng 920 processor (ARMv8 AArch64)
   * 512GB main memory: 16x 32GB DDR4 2933 MHz RDIMM
   * Local storage: 2x 960GB SSD SATA 6Gb/s
   * Data storage: 4 x 960GB SSD SATA 6Gb/s
   * GPU: 5x Atlas 300 AI Inference Card; 32GB; PCIe3.0 x16

The nodes are connected via a non-blocking EDR Infiniband 100GBit/s fabric for high bandwidth, low latency communication as well as an Ethernet network for deployment and management.

Open source software is used for the cluster software environment. The cluster is built on Rocky Linux 8 with SLURM as a job scheduler. Lustre in combination with ZFS is used to provide a high-performance parallel filesystem to the users. EasyBuild is used extensively to install and configure software packages: GCC, OpenMPI, Python, SciPy and others. The AI software stack is built on top of Huawei’s CANN (Compute Architecture for Neural Networks) toolkit, which will support AI frameworks like TensorFlow, PyTorch, ONNX and Mindspore.

For any queries, please contact karthee dot Sivalingam at huawei.com

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   access.rst
   usage.rst
   mindspore_page.rst
   pytorch_guide.rst
   tensorflow_guide.rst
   atc_guide.rst
   faq_and_links.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
