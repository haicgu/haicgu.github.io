==============================================================================
ATC Environment Setup & User Guide
==============================================================================
1 - Model Conversion with ATC
==============================================================================

You can use ATC_ (Ascend Tensor Compiler) to convert network models trained on open source frameworks to offline models supported by Ascend AI Processor.

.. _ATC: https://support.huaweicloud.com/intl/en-us/ti-atc-A200_3000/ATC_Tool_Instructions.pdf

1.1 - Module Load
-----------------------

ATC tool can work with all CANN modules and does not need to load different modules apart from them.

1.2 - Model Conversion
-----------------------

Conversion Script::

    
    module cat <<EOF > batchscript.sh #Declare execute path
    #!/bin/bash
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00  . 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    atc <use necessery flags>
    EOF
    sbatch batchscript.sh
    

1.3 - Example Usage
-----------------------

Environment Preperation::

    
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/aipp_nv12.cfg
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3.prototxt
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3.caffemodel
    

Training Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash > 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 > npu-smi info
    atc --model=yolov3.prototxt --weight=yolov3.caffemodel --framework=0 --output=yolov3_yuv --soc_version=Ascend310 --insert_op_conf=aipp_nv12.cfg
    EOF
    
Run the script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
    
For further examples: ATC_ .

.. _ATC: https://support.huaweicloud.com/intl/en-us/ti-atc-A200_3000/ATC_Tool_Instructions.pdf

2 - Inference
==============================================================================

2.1 - Offline Inference
-----------------------

Offline inference means, running an operation with model which translated with ATC on inference devices. All process can be executed by using ACL (Ascend Computing Language) supporting Python and C++ languages developed by Huawei.

2.1.1 - C++ (ACL)
-----------------------
This ACL-document_ provides guidance for developing deep neural network (DNN) apps based on existing models by using C language APIs provided by the Ascend Computing Language (AscendCL), for such purposes as target recognition and image classification.

.. _ACL-document: https://www.hiascend.com/document/detail/en/canncommercial/504/inferapplicationdev/aclcppdevg/aclcppdevg_000000.html 

2.1.3 -  Module Load
-----------------------
Environment preparation::
    
    module load GCC/9.5.0 OpenMPI CANN-Toolkit OpenCV
    

2.1.4 - Model Inference
-----------------------
Inference Script::
    
    module cat <<EOF > batchscript.sh
    #!/bin/bash
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00
    #SBATCH --ntasks=1
    #SBATCH --nodes=1 
    npu-smi info
    atc <use necessery flags>
    <Inference command>
    EOF

    sbatch batchscript.sh

    
2.1.4 - Example Usage
-----------------------
- Example Repo: ACL_

.. _ACL:  https://gitee.com/ktuna/acl_multi_stream_inference

Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI CANN-Toolkit OpenCV
    git clone https://gitee.com/ktuna/acl_multi_stream_inference.git
    cd acl_multi_stream_inference
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/aipp_nv12.cfg
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3.prototxt
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3.caffemodel
    
Inference Script::
    
    module cat <<EOF > batchscript.sh >
    #!/bin/bash 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1
    #SBATCH --nodes=1 
    npu-smi info
    atc --model=yolov3.prototxt --weight=yolov3.caffemodel --framework=0 --output==/data/model/yolov3 --soc_version=Ascend310 --insert_op_conf=aipp_nv12.cfg ./build.sh
    cd dist
    ./main.sh
    EOF
    
Run the Script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
    

 2.1.5 - Python (PyACL)
-----------------------

This PyACL-document_ provides guidance for developers to develop deep neural network (DNN) applications for purposes including target recognition and image classification based on existing models and Python APIs provided by Python Ascend Computing Language (pyACL).

.. _PyACL-document: https://www.hiascend.com/document/detail/en/canncommercial/504/inferapplicationdev/aclpythondevg/aclpythondevg_0000.html

2.1.6 - Module Load
-----------------------
Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI CANN-Toolkit 
    

2.1.7 - Module Inference
-----------------------
Inference Script::
    
    module cat <<EOF > batchscript.sh
    #!/bin/bash
    #SBATCH --partition=a800-9000
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1
    #SBATCH --nodes=1 
    npu-smi info
    atc <use necessery flags>
    <Inference command>
    EOF

    sbatch batchscript.sh
    

2.1.8 - Example Usage:
-----------------------
- Example Repo: YOLOv3_

.. _YOLOv3: https://gitee.com/ascend/samples/tree/master/python/level2_simple_inference/2_object_detection/YOLOV3_coco_detection_picture

Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI CANN-Toolkit 
    git clone https://gitee.com/ascend/samples.git 
    cd samples/python/level2_simple_inference/
    cd 2_object_detection/YOLOV3_coco_detection_picture/$ wget https://modelzoo-train-atc.obs.cn-north4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/aipp_nv12.cfg
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3.prototxt
    wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3.caffemodel
    
Inference Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    atc --model=yolov3.prototxt --weight=yolov3.caffemodel --framework=0 --output==/data/model/yolov3 --soc_version=Ascend310 --insert_op_conf=aipp_nv12.cfg
    python3 object_detect.py ../data/
    EOF
    
For further examples: Tensorflow_ 2 Gitee Repository  , PyACL_ Repository , Example C++ Inference_ Repository 

.. _Tensorflow: https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in
.. _PyACL: https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800
.. _Inference: https://gitee.com/ktuna/acl_multi_stream_inference

