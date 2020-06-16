# EXERCISE-USING-INTEL-DEVCLOUD

Requesting a device on Intel's DevCloud and loading a model, and running inference on an image.

In this exercise, you will do the following:

1. Write a Python script to load a model and run inference 10 times on a CPU on Intel's DevCloud.
    . Calculate the time it takes to load the model.
    . Calculate the time it takes to run inference 10 times.
2. Write a shell script to submit a job to Intel's DevCloud.
3. Submit a job using <code>qsub</code> on the <strong>IEI Tank-870</strong> edge node with an <strong>Intel Xeon E3 1268L</strong>
4. Run <code>liveQStat</code> to view the status of your submitted job.
5. Retrieve the results from your job.
6. View the results.
    
<pre><code>%env PATH=/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/intel_devcloud_support
import os
import sys
sys.path.insert(0, os.path.abspath('/opt/intel_devcloud_support'))
sys.path.insert(0, os.path.abspath('/opt/intel'))</code></pre>

# The Model

We will be using the `vehicle-license-plate-detection-barrier-0106` model for this exercise. Remember that to run a model on the CPU, we need to use `FP32` as the model precision.

The model has already been downloaded for you in the `/data/models/intel` directory on Intel's DevCloud. We will be using the following filepath during the job submission in **Step 3**:

> **/data/models/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106**

We will be running inference on an image of a car. The path to the image is `/data/resources/car.png`

# Step 1: Creating a Python Script

The first step is to create a Python script that you can use to load the model and perform inference. We'll use the <code>%%writefile</code> magic to create a Python file called <code>inference_cpu_model.py</code>. In the next cell, you will need to complete the <code>TODO</code> items for this Python script.

<code>TODO</code> items:

1. Load the model
2. Prepare the model for inference (create an input dictionary)
3. Run inference 10 times in a loop

<pre><code>%%writefile inference_cpu_model.py

import time
import numpy as np
import cv2
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start=time.time()
    model=IENetwork(model_structure, model_weights)

    core = IECore()
    net = core.load_network(network=model, device_name='CPU', num_requests=1)
    print(f"Time taken to load model = {time.time()-start} seconds")
    
    # Get the name of the input node
    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    input_img=cv2.imread('/data/resources/car.png')
    input_img=cv2.resize(input_img, (300,300), interpolation = cv2.INTER_AREA)
    input_img=np.moveaxis(input_img, -1, 0)

    # Running Inference in a loop on the same image
    input_dict={input_name:input_img}

    start=time.time()
    for _ in range(10):
        net.infer(input_dict)
    
    print(f"Time Taken to run 10 Infernce on CPU is = {time.time()-start} seconds")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    
    args=parser.parse_args() 
    main(args)</code></pre>

