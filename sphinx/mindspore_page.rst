==============================================================================
Ascend- AI Cluster Mindspore Environment Guide
==============================================================================


1 - MindSpore
==============================================================================

Through community cooperation, this open Al framework best matches with Ascend processors and supports multi-processor architectures for all scenarios. It brings data scientists, algorithm engineers, and developers with friendly development, efficient running, and flexible deployment, and boosts the development of the Al software and hardware ecosystem.

Official Page: Mindspore_

.. _Mindspore: https://www.mindspore.cn/en


2 - MindSpore Environment Module Load
==============================================================================

2.1 - Predownloaded Module Load
-----------------------------------

Command::
	
    module load GCC/9.5.0 OpenMPI MindSpore

Quick Check::

    srun -p a800-9000 python3 -c 'from mindspore import context;context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")'



3 - Code Examples for Testing
==============================================================================


3.1 - Example 1
-----------------------

Code ::

    srun -p a800-9000 python3 -c "import mindspore;mindspore.run_check()"

Output ::

    MindSpore version: 1.6.2 
    The result of multiplication calculation is correct, MindSpore has been installed successfully! 


3.2 - Example 2
-----------------------

Code ::

    import numpy as np 
    import mindspore as ms 
    import mindspore.ops as ops 	 
    
    ms.set_context(device_target="Ascend") 
    x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32)) 
    y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32)) 
    print(ops.add(x, y))


Output ::

    [[[[2. 2. 2. 2.] [2. 2. 2. 2.] [2. 2. 2. 2.]] [[2. 2. 2. 2.] [2. 2. 2. 2.] [2. 2. 2. 2.]] [[2. 2. 2. 2.] [2. 2. 2. 2.] [2. 2. 2. 2.]]]] 


For more examples, check MindSpore_Documentation_ 1.6.2 Documentation

.. _MindSpore_Documentation: https://www.mindspore.cn/tutorials/en/r1.6/index.html



