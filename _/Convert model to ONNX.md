# Convert model to ONNX

Didn't work, need to try again using a docker container

References:

- <https://github.com/bmitu/mmdeploy/blob/master/docs/en/get_started.md>
- https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#maclearn-net-repo-install
- https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
- https://developer.nvidia.com/nvidia-tensorrt-8x-download

~~nvidia-smi~~
~~--> CUDA Version: 12.0~~

nvcc --version
--> 11.5

sudo apt install nvidia-cuda-toolkit

conda create --name mmdeploy python=3.9 -y
conda activate mmdeploy

conda install pytorch torchvision cudatoolkit=11.5 -c pytorch -c conda-forge -y

pip install -U openmim
mim install mmcv-full


This didn't work: ModuleNotFoundError: No module named 'torch' (even if pythorch was just installed)

----

<!-- python3.10 venv -->

<!-- pip install torch torchvision cudatoolkit==11.5 -->

Use openmmlab conda env instead - but with mmcv==1.7, original openmmlab env needs to be rebuilt


# install MMDeploy
wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.12.0/mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1.tar.gz
tar -zxvf mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1.tar.gz
cd mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1
pip install dist/mmdeploy-0.12.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.12.0-cp38-none-linux_x86_64.whl
cd ..
# install inference engine: ONNX Runtime
pip install onnxruntime==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH


pip install -v -e .

wget -P checkpoints https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth



!!! https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud

conda install pytorch torchvision torchaudio pytorch-cuda=11.5 -c pytorch -c nvidia

didn't work, no 11.5 available

---

Trying pip

sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.9-dev
sudo apt install python3.9-distutils

python3.9 -m venv .venv --without-pip

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py


```shell
sudo apt install nvidia-cuda-toolkit

nvcc --version
--> 11.5
```

Activate venv in mmdeploy

```shell
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu115
pip install -U openmim
mim install mmcv-full

```

### Install CUDNN

wget https://developer.download.nvidia.com/compute/cudnn/secure/8.8.0/local_installers/11.8/cudnn-linux-x86_64-8.8.0.121_cuda11-archive.tar.xz?7nULOX85gPRamn02bS3X-o-5daYPreylPPrgClOb3rRqpH3FZSLJG37gydIH4iSvKvrjhARoKkK-Wzy8FS45scax5J_9_IE4IM9os1DhFb3QJO0aPJsgHTZ0j7guHU5gqJ__WBKM04Z-Sinn_XNu_1B8gnnzrP3wcaXedWvMuIa29MGsiTO0GzNQGIyVoQYZVGLSHHaNLk97zrDiaIwXypHO&t=eyJscyI6IndlYnNpdGUiLCJsc2QiOiJkZXZlbG9wZXIubnZpZGlhLmNvbS9udmlkaWEtdHJpdG9uLWluZmVyZW5jZS1zZXJ2ZXIifQ==

tar -xvf cudnn-linux-x86_64-8.8.0.121_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.8.0.121_cuda11-archive/include/cudnn*.h /usr/lib/cuda/include 
sudo cp -P cudnn-linux-x86_64-8.8.0.121_cuda11-archive/lib/libcudnn* /usr/lib/cuda/lib64
sudo chmod a+r /usr/lib/cuda/include/cudnn*.h /usr/lib/cuda/lib64/libcudnn*

## Install TensorRT

From tar archive:

```shell
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.5.3/tars/TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz?Tt-xjV83ifwpgQf10cNHMQP158jAgSbZ-dA1OYJLEVKfF2nqqikfW7ZsgLKJILjSkcTwHALogKwfa6NNkt0SCmL-Qk1awLo637YhIogPUr4H2Nd--Q0IzuvIF4Fb5lc3THtjvZeomcLzOPWchhpKknTbh5dZq50mhWXrAp7k7euiZ-KnvIuJaSxEbB8M9Fr3iMJw81Gq-Tvvd2Y0yxXQNqz_ec1rDkvl4fYmtXj-&t=eyJscyI6IndlYnNpdGUiLCJsc2QiOiJkZXZlbG9wZXIubnZpZGlhLmNvbS9udmlkaWEtdHJpdG9uLWluZmVyZW5jZS1zZXJ2ZXIifQ== 

(rename to TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz)

tar -xzvf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-8.5.3.1/lib

cd TensorRT-8.5.3.1/python
python3 -m pip install tensorrt-8.5.3.1-cp39-none-linux_x86_64.whl

cd ../graphsurgeon/
python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl

cd ../onnx_graphsurgeon/
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

dpkg -l | grep TensorRT
```

From deb package:

```shell
wget -O nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.5.3/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb?FnKOwB4T8RpfF24whl4BeTo6RJMcSOPoF1aobI5NglURuvFei9xeIDWzerkxbevnO2gBzQv7V7rwR9BCDv5Eg5r36vm6T3tAH89z9VRho9kdwJrMz_7Iy-VQw2a2ZViPISOt4Fxh9_S1PcOS2-AV5B02DCNYgYgTl-FfX-b6VNrQggpi4mPDRQ0jK0ZBSQR39wi3lQrFKl-xp8lavtjUNVHomR161IajE07qFx_mFhb7XFXHCS3rQq8aEg==&t=eyJscyI6IndlYnNpdGUiLCJsc2QiOiJkZXZlbG9wZXIubnZpZGlhLmNvbS9udmlkaWEtdHJpdG9uLWluZmVyZW5jZS1zZXJ2ZXIifQ==

os="ubuntu2204"
tag="8.5.3-cuda-11.8"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/


sudo apt install nvidia-cudnn

sudo apt-get update
sudo apt-get install tensorrt


```


## Install MMDeploy

From OpenMMLab:

wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.12.0/mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1.tar.gz
tar -zxvf mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1.tar.gz
cd mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1
pip install dist/mmdeploy-0.12.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.12.0-cp39-none-linux_x86_64.whl
cd ..

# Install inference engine: ONNX Runtime

pip install onnxruntime==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

## Convert Model

```
cd mmdetection
pip install -v -e .
cd ..

# download Faster R-CNN checkpoint
wget -P checkpoints https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# run the command to start model conversion
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    mmdetection/demo/demo.jpg \
    --work-dir mmdeploy_model/faster-rcnn \
    --device cuda \
    --dump-info
```



