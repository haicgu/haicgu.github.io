==============================================================================
Ascend- AI Cluster Quick Environment Setup & User Guide
==============================================================================


1 - Model Conversion with ATC
==============================================================================

You can use `ATC (Ascend Tensor Compiler) <https://support.huaweicloud.com/intl/en-us/ti-atc-A200_3000/ATC_Tool_Instructions.pdf>` to convert network models trained on open source frameworks to offline models supported by Ascend AI Processor.

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

2 - Training
==============================================================================
Training a model simply means learning (determining) good values for all the weights and the bias from labeled examples. Huawei Ascend 910 Chips mainly developed for high performance training but also supports inference.

2.1 - Pytorch-v1.5.0
-----------------------
The modules, model and so on information used during the PyTorch-v1.5.0 training performed on the specified cluster are given below.

2.1.1.1 - Module Load
Environment preparation::
    
    module load GCC/9.5.0 OpenMPI PyTorch-CANN/1.5.0
    

Note::
- If you want to train model with mixed precision, you also need to load Apex module
    
    module load apex
    


2.1.1.2 Model Training
-----------------------
Training Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00  
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    #Displays avaliablity of NPU's > npu-smi info
    <Training command>
    EOF

    sbatch batchscript.sh

    

2.1.1.3 Example Usage
-----------------------
- Example Repo : LeNet_ . 
 
.. _LeNet: https://gitee.com/tianyu__zhou/pytorch_lenet_on_npu

Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI PyTorch-CANN/1.5.0 apex
    git clone https://gitee.com/tianyu__zhou/pytorch_lenet_on_npu.git
    cd pytorch_lenet_on_npu
    
Training Script::
    
    module cat <<EOF > batchscript.sh
    #!/bin/bash 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1
    #SBATCH --nodes=1 
    npu-smi info
    export RANK_SIZE=1
    python3 train_npu.py --epochs 10 --batch-size 64 --device_id 0
    EOF
    
Run the Script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
    

2.2 - Tensorflow-v1.15
-----------------------

2.2.1.4 - Module Load
-----------------------
Environment Preperation::
    bash
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/1.15.0
    

2.2.1.5 - Model Training
-----------------------
Training Script::
    
    module cat <<EOF > batchscript.sh
    #!/bin/bash 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00  
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    <Training command>
    EOF

    sbatch batchscript.sh

    

2.2.1.6 - Example Usage
-----------------------

- Example Repo : LeNet_ . 
.. _LeNet: https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/nlp/LeNet_ID0127_for_TensorFlow

Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/1.15.0
    git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
    cd ModelZoo-TensorFlow/TensorFlow/built-in/nlp/LeNet_ID0127_for_TensorFlow/
    
Training Script::
    
    module cat <<EOF > batchscript.sh
    #!/bin/bash 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    python LeNet.py ---data_path <data_path>
    EOF
    
Run the script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
    
2.3 - Tensorflow-v.2.4
-----------------------

2.3.1.7 - Module Load
-----------------------
Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/2.4.1
    

2.3.1.8 Model Training
-----------------------
Training Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash
    #SBATCH --partition=a800-9000
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    <Training command>
    EOF


    sbatch batchscript.sh

    

2.3.1.9 - Example Usage
-----------------------
- Example Usage: BYOL_ .
.. _BYOL: https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/cv/image_classification/BYOL_ID0721_for_TensorFlow2.X

Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/2.4.1
    git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
    cd ModelZoo-TensorFlow/TensorFlow2/built-in/cv/image_classification/BYOL_ID0721_for_TensorFlow2.X
    
Training Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    python retraining.py --encoder resnet18 --num_epochs 1 --batch_size 64
    EOF
    
Run the script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
    
For Further Examples: TensorFlow2_ . 
.. _TensorFlow2:  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/cv/image_classification/BYOL_ID0721_for_TensorFlow2.X

3 - Inference
==============================================================================

Model inference is the process of using a trained model to infer a result from live data. Ascend 310 chips supports only inference. 


Note::
 - A310 chips are developed much smaller than A910 chips to bring inference solutions in real life more easily with lesser power consumption with more affordable price. You can discover more from the `link <https://www.hiascend.com/hardware/product>`


3.1 - Offline Inference
-----------------------

Offline inference means, running an operation with model which translated with ATC on inference devices. All process can be executed by using ACL (Ascend Computing Language) supporting Python and C++ languages developed by Huawei.

3.1.1 - C++ (ACL)
-----------------------
This document_ provides guidance for developing deep neural network (DNN) apps based on existing models by using C language APIs provided by the Ascend Computing Language (AscendCL), for such purposes as target recognition and image classification.

.. _document https://www.hiascend.com/document/detail/en/canncommercial/504/inferapplicationdev/aclcppdevg/aclcppdevg_000000.html 

3.1.1.1 -  Module Load
-----------------------
Environment preparation::
    
    module load GCC/9.5.0 OpenMPI CANN-Toolkit OpenCV
    

3.1.1.2 - Model Inference
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

    
3.1.1.3 - Example Usage
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
    

 3.1.2 - Python (PyACL)
-----------------------

This `document <https://www.hiascend.com/document/detail/en/canncommercial/504/inferapplicationdev/aclpythondevg/aclpythondevg_0000.html>` provides guidance for developers to develop deep neural network (DNN) applications for purposes including target recognition and image classification based on existing models and Python APIs provided by Python Ascend Computing Language (pyACL).

3.1.2.4 - Module Load
-----------------------
Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI CANN-Toolkit 
    

3.1.2.5 - Module Inference
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
    

3.1.2.6 - Example Usage:
-----------------------
- Example Repo: `YOLOv3 <https://gitee.com/ascend/samples/tree/master/python/level2_simple_inference/2_object_detection/YOLOV3_coco_detection_picture>`

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
    
For further examples: `Tensorflow2 Gitee Repository <https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in>` , `PyACL Repository <https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800>`, `Example C++ Inference Repository <https://gitee.com/ktuna/acl_multi_stream_inference>`

3.2 - Online Inference
-----------------------


Online inference means, running an operation without converting model. It supports to inference Tensorflow, Pytorch and Mindspore models in original form. While Huawei A310 chips supports Tensorflow and Mindspore models for online inference, A910 chips supports Tensorflow, Pytorch and Mindspore models.

3.2.1 - Pytorch-v1.5.0
-----------------------
The modules, model and so on information used during the PyTorch-v1.5.0 Online Inference performed on the specified cluster are given below.

3.2.1.7 - Module Load
-----------------------
Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI PyTorch-CANN/1.5.0
    
3.2.1.8 - Model Inference
-----------------------
Inference Script::
    
    module cat <<EOF > batchscript.sh
    #!/bin/bash
    #SBATCH --partition=a800-9000
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    <Inference command>
    EOF

    sbatch batchscript.sh
    
3.2.1.9 - Example Usage
-----------------------
Example Repo: `ResNet-50 <https://gitee.com/ascend/pytorch/blob/master/docs/en/PyTorch%20Online%20Inference%20Guide/PyTorch%20Online%20Inference%20Guide.md#sample-code>`
Environmental Preperation:::
    
    module load GCC/9.5.0 OpenMPI PyTorch-CANN/1.5.0  
    
- The code we will use for inference is in the readme. For this, we need to open a new python file and copy the code there.
    
    vim resnet50_infer_for_pytorch.py  #paste the code in here
    
- Visit `Ascend ModelZoo <https://www.hiascend.com/software/modelzoo>` and click Download Model to download a pre-trained ResNet-50 model.


Inference Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    python3 resnet50_infer_for_pytorch.py --data ./data/ --npu 0 --epochs 90 --resume ./ResNet50_for_Pytorch_1.4_model/resnet50_pytorch_1.4.pth.tar
    EOF
    
Run the script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
        
 3.2.2 - TensorFlow-v1.15
-----------------------
The modules, model and so on information used during the Tensorflow-v1.15 Online Inference performed on the specified cluster are given below.

3.2.2.10 - Module Load
-----------------------
Environmental Preperation::
    
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/1.15.0
    

3.2.2.11 - Model Inference
-----------------------
Inference Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash
    #SBATCH --partition=a800-9000  
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    <Inference command>
    EOF

    sbatch batchscript.sh
    

3.2.2.12 - Example Usage
-----------------------
- Example Repo: `FAceNet <https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800/facenet>`
Environmental Preperation::
    
    module load GCC/9.5.0 OpenMPI Tensorlfow-CANN/1.15.0 
    git clone https://gitee.com/tianyu__zhou/pyacl_samples.git
    cd ./pyacl_samples/facenet
    
- Visit `FaceNet Original Repo <https://github.com/davidsandberg/facenet>` and click Download Model to download a pre-trained Facenet model.
Inference Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash 
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    python3 resnet50_infer_for_pytorch.py --data ./data/ --npu 0 --epochs 90 --resume ./ResNet50_for_Pytorch_1.4_model/resnet50_pytorch_1.4.pth.tar>EOF
    

Run the Script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
    

4 - FAQ's
==============================================================================

4.1 -  No module named 'torch_npu'
-----------------------
If pytorch is installed correctly, The error can be fixed by removing the```import torch_npu``` and writing ```torch.npu()``` instead of ```torch_npu.npu()``` .

4.2 - Val folder path error when giving data flag
-----------------------
While creating the resnet50_infer_for_pytorch.py file, you need to edit the data path part. 
If you comment out the line ```valdir = os.path.join(args.data, 'val')``` and write ```args.data``` instead of the ```valdir``` assigned to the ```val_loader``` variable, the problem will go away.

Output::

    # =========================================================================
    # Initialize the database.
    # =========================================================================
    # Load and preprocess image data.
    #valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

4.3 - How can I convert my model?
-----------------------
You can use ATC to convert your trained model to om (offline model). For more information click `here <https://support.huawei.com/enterprise/en/doc/EDOC1100192457/2a40d134/atc-tool>`.

4.4 - How can I see the status of NPU's?
-----------------------
You can see by writing ```srun -p a800-9000 npu-smi info``` .

4.5 - I am trying to train model from Ascend-Modelzoo but I can't find the example data.
-----------------------
Most of the models don't have their own built in data. So that you should provide data to train your model. But some models requires different formats (e.g. .npz, .tfrecords etc.)

4.6 - I can't run my Tensorflow Script!
-----------------------
Be careful. There are two different TF module to use. One supports 1.15 and other supports 2.4. So that try to change between ```module load GCC/9.5.0 OpenMPI TensorFlow-CANN/2.4.1``` and ```module load GCC/9.5.0 OpenMPI TensorFlow-CANN/1.15.0```

4.7 - My module can not find OpenCV or any other package!
-----------------------
Add OpenCV or anything you want at the end of your module command like ```module load GCC/9.5.0 OpenMPI CANN-Toolkit OpenCV``` 

5 - Useful Links
==============================================================================
-	**Ascend official website (includes all information about ascend platform)**
https://www.hiascend.com/ 

-	**Ascend ModelZoo website (includes all models available for ascend)**
https://www.hiascend.com/software/modelzoo 

-	**AscendHub website (includes docker images based on specific frameworks)**
https://ascendhub.huawei.com/#/index 

-	**Ascend developer zone (forum)**
https://forum.huawei.com/enterprise/en/forum-100504.html 

-	**Huawei cloud-bbs**
https://bbs.huaweicloud.com/

-	**Ascend official gitee page (includes sample repositories about ascend platform usage)**
https://gitee.com/ascend 

-	**pyACL sample gitee page (papered by FAEs)**
https://gitee.com/tianyu__zhou/pyacl_samples

-	**Huawei support ascend computing (includes all product information)**
https://support.huawei.com/enterprise/en/category/ascend-computing-pid-1557196528909?submodel=doc
