Here is the summary what we need to do for running the code:

-- Install Python version 3.x (using Anaconda we got errors because of pip installation, normally it should work with Anaconda)

-- (Optional) Create virtual environment e.g. pytorch (--> conda create -n yourenvname)

-- Install PyTorch using: https://pytorch.org/get-started/locally/

-- Install tqdm:  pip install tqdm

-- install cv2:  pip install opencv-python

-- install PIL: pip install Pillow

After all these installation it was giving error because of Cuda. 
If Cuda installation is needed, it should be done before all of these steps 
(it could be done after Python installation but should be definitely before pytorch installation!)

-- Ensure CUDA 8.0 is installed
-- Ensure cudnn is installed (https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

To test the code:
- Open Anaconda Prompt
- Activate virtual environment (optional --> activate yourenvname)
- Navigate to clean_u_net\u_net_test
- Run main.py script (--> python main.py)