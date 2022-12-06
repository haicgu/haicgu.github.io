==============================================================================
FAQ's and Useful Links for Ascend, ATC, Pytorch, Tensorflow
==============================================================================


1 - FAQ's
==============================================================================

1.1 -  No module named 'torch_npu'
-----------------------
If pytorch is installed correctly, The error can be fixed by removing the```import torch_npu``` and writing ```torch.npu()``` instead of ```torch_npu.npu()``` .

1.2 - Val folder path error when giving data flag
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

1.3 - How can I convert my model?
-----------------------
You can use ATC to convert your trained model to om (offline model). For more information click here_.

.. _here: https://support.huawei.com/enterprise/en/doc/EDOC1100192457/2a40d134/atc-tool

1.4 - How can I see the status of NPU's?
-----------------------
You can see by writing ``srun -p a800-9000 npu-smi info`` .

1.5 - I am trying to train model from Ascend-Modelzoo but I can't find the example data.
-----------------------
Most of the models don't have their own built in data. So that you should provide data to train your model. But some models requires different formats (e.g. .npz, .tfrecords etc.)

1.6 - I can't run my Tensorflow Script!
-----------------------
Be careful. There are two different TF module to use. One supports 1.15 and other supports 2.4. So that try to change between ``module load GCC/9.5.0 OpenMPI TensorFlow-CANN/2.4.1`` and ``module load GCC/9.5.0 OpenMPI TensorFlow-CANN/1.15.0``

1.7 - My module can not find OpenCV or any other package!
-----------------------
Add OpenCV or anything you want at the end of your module command like ``module load GCC/9.5.0 OpenMPI CANN-Toolkit OpenCV``

2 - Useful Links
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
