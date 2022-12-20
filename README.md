# Autopilot2PytorchSteering

Pytorch implementation of Autopilot2 to determine the steering angle
Input: RGB
Output: Steering Angle

# Autopilot

[A simple self-driving car module for humans](https://github.com/akshaybahadur21/Autopilot)   

# Code Requirements
1. OS is Ubuntu 18.04
2. Pythen version: 3.6
3. Dependency
For Anaconda
```python
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge h5py opencv scikit-learn matplotlib pandas scipy imageio visdom jsonpatch
```

# Test with weighted model

With correct angle

`python test.py --model "./models/model_10000.pkl" --input "./testcase/2.jpg" --angle -0.019373154`

Without target angle

`python test.py --model "./models/model_10000.pkl" --input "./testcase/1.jpg"`

# Download Dataset

Download the dataset at [here](https://github.com/SullyChen/driving-datasets) and extract into the repository folder `driving_dataset`


# Launch a visdom.server
> For training visualization

Open a terminal, run `python -m visdom.server` (in your conda virtual env)
Pay attension to you 8097 port, it needs to be occupied by visdom.

# Models

`./models/model_10000.pkl`

`./models/model_100.pkl`

# Train

`python train.py`


