==============================================================================
Pytorch Environment Setup & User Guide
==============================================================================

1 - Training
==============================================================================
Training a model simply means learning (determining) good values for all the weights and the bias from labeled examples. Huawei Ascend 910 Chips mainly developed for high performance training but also supports inference.

1.1 - Pytorch-v1.5.0
-----------------------
The modules, model and so on information used during the PyTorch-v1.5.0 training performed on the specified cluster are given below.

1.1.1 - Module Load
Environment preparation::
    
    module load GCC/9.5.0 OpenMPI PyTorch-CANN/1.5.0
    

Note: If you want to train model with mixed precision, you also need to load Apex module ::
    
    module load apex


1.1.2 Model Training
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

    

1.1.3 Example Usage
-----------------------
- Example Repo : LENET_
.. _LENET: https://gitee.com/tianyu__zhou/pytorch_lenet_on_npu

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

2 - Inference
==============================================================================

Model inference is the process of using a trained model to infer a result from live data. Ascend 310 chips supports only inference. 

Note::
 - A310 chips are developed much smaller than A910 chips to bring inference solutions in real life more easily with lesser power consumption with more affordable price. You can discover more from the link_ .
 
.. _link: https://www.hiascend.com/hardware/product

2.1 - Online Inference
-----------------------


Online inference means, running an operation without converting model. It supports to inference Tensorflow, Pytorch and Mindspore models in original form. While Huawei A310 chips supports Tensorflow and Mindspore models for online inference, A910 chips supports Tensorflow, Pytorch and Mindspore models.

2.1.1 - Pytorch-v1.5.0
-----------------------
The modules, model and so on information used during the PyTorch-v1.5.0 Online Inference performed on the specified cluster are given below.

2.1.2 - Module Load
-----------------------
Environment Preperation::
    
    module load GCC/9.5.0 OpenMPI PyTorch-CANN/1.5.0
    
2.1.3 - Model Inference
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
    
2.1.4 - Example Usage
-----------------------
Example Repo: ResNet_-50 

.. _ResNet: https://gitee.com/ascend/pytorch/blob/master/docs/en/PyTorch%20Online%20Inference%20Guide/PyTorch%20Online%20Inference%20Guide.md#sample-code
Environmental Preperation:::
    
    module load GCC/9.5.0 OpenMPI PyTorch-CANN/1.5.0  
    
- The code we will use for inference is in the readme. For this, we need to open a new python file and copy the code there.
    
    vim resnet50_infer_for_pytorch.py  #paste the code in here
    
- Visit Ascend ModelZoo_  and click Download Model to download a pre-trained ResNet-50 model.

.. _ModelZoo: https://www.hiascend.com/software/modelzoo

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

