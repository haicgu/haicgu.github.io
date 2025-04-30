Filesystems on HAICGU
=====================

There are multiple filesystems accessible to users on HAICGU

.. list-table:: Filesystems on HAICGU
   :widths: 40 10 20 10 20 10
   :header-rows: 1

   * - Mount point
     - Backend
     - Available
     - Capacity
     - Network
     - Performance
   * - /home
     - guoehi (NFS)
     - all
     - 8TB
     - 1Gb ETH (ml01, ml02, dev, cn{01-09, 19-28}); 100Gb IB (cn{10-18})
     - TODO
   * - /mnt/lustre-storage
     - guoehi-io0{1,2} (Lustre)
     - all - guoehi
     - 36TB
     - Network: cn{10-18} 1Gbit ; ml01, ml02, dev, cn{01-09, 19-28} 100Gbit
     - TODO
   * - /mnt/OSP-data1
     - `OSP 9950 <https://carrier.huawei.com/en/products/it-new/oceanstorpacific/oceanstor-pacific-9950>`_ (NFS)
     - dev, cn{01-09, 19-28}
     - 36TB
     - 100 Gb ETH
     - TODO
   * - /mnt/dev-lscratch
     - 2xMZ7KH960HAJR (ZFS)
     - dev
     - 1.8TB
     - -
     - TODO
   * - /mnt/ml01-sscratch
     - 2xMZ7KH960HAJR (ZFS)
     - ml01
     - 1.8TB
     - -
     - TODO
   * - /mnt/ml01-nscratch
     - 4xMZWLL3T2HAJQ-00005 (ZFS)
     - ml01
     - 12 TB
     - -
     - TODO
   * - /mnt/ml02-sscratch
     - 6xMZ7KH960HAJR (ZFS)
     - ml01
     - 5.2TB
     - -
     - TODO

S3 Access
=========

HAICGU has an S3 Access Point. Please contact haicgu@fias.uni-frankfurt.de for access.

