==============================================================================
Tesnorflow Environment Setup & User Guide
==============================================================================

1 - Training
==============================================================================
Training a model simply means learning (determining) good values for all the weights and the bias from labeled examples. Huawei Ascend 910 Chips mainly developed for high performance training but also supports inference.


1.1 - Tensorflow-v1.15
-----------------------

1.1.2 - Module Load
-----------------------
Environment Preperation::
    bash
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/1.15.0
    

1.1.3 - Model Training
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

    

1.1.4 - Example Usage
-----------------------

- Example Repo : LeNet_git_ . 
.. _LeNet_git: https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/nlp/LeNet_ID0127_for_TensorFlow

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
    
1.2 - Tensorflow-v.2.4
-----------------------

1.2.1 - Module Load
-----------------------
Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/2.4.1
    

1.2.2 Model Training
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

    

1.2.3 - Example Usage
-----------------------
- Example Usage: LeNetMNIST_ .
.. _LeNetMNIST: https://gitee.com/tianyu__zhou/tf2_lenet_on_npu/blob/master/train_npu.py

Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/2.4.1
    git clone https://gitee.com/tianyu__zhou/tf2_lenet_on_npu.git
    cd tf2_lenet_on_npu
    
Training Script::
    
    module cat <<EOF > batchscript.sh 
    #!/bin/bash
    #SBATCH --partition=a800-9000 
    #SBATCH --time=00:10:00 
    #SBATCH --ntasks=1 
    #SBATCH --nodes=1 
    npu-smi info
    python3 train_npu.py
    EOF
    
Run the script::
    
    sbatch batchscript.sh
    >>> Submitted batch job 1079
    cat slurm-1079.out
    >>> [...]
    
For Further Examples: TensorFlow2_ .

.. _TensorFlow2:  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/cv/image_classification/BYOL_ID0721_for_TensorFlow2.X

2 - Inference
==============================================================================

Model inference is the process of using a trained model to infer a result from live data. Ascend 310 chips supports only inference. 

Note::
 - A310 chips are developed much smaller than A910 chips to bring inference solutions in real life more easily with lesser power consumption with more affordable price. You can discover more from the link_ .
 
.. _link: https://www.hiascend.com/hardware/product

2.1 - Online Inference
-----------------------


Online inference means, running an operation without converting model. It supports to inference Tensorflow, Pytorch and Mindspore models in original form. While Huawei A310 chips supports Tensorflow and Mindspore models for online inference, A910 chips supports Tensorflow, Pytorch and Mindspore models.


2.2 - TensorFlow-v1.15
-----------------------
The modules, model and so on information used during the Tensorflow-v1.15 Online Inference performed on the specified cluster are given below.

2.2.1 - Module Load
-----------------------
Environmental Preperation::
    
    module load GCC/9.5.0 OpenMPI TensorFlow-CANN/1.15.0
    

2.2.2 - Model Inference
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
    

2.2.3 - Example Usage
-----------------------
- Example Repo:`FaceNet_ 

.. _FaceNet: https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800/facenet

Environmental Preperation::
    
    module load GCC/9.5.0 OpenMPI Tensorlfow-CANN/1.15.0 
    git clone https://gitee.com/tianyu__zhou/pyacl_samples.git
    cd ./pyacl_samples/facenet
    
- Visit FaceNet Original Repository_  and click Download Model to download a pre-trained Facenet model.

.. _Repository: https://github.com/davidsandberg/facenet

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
