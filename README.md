# Webcam-Mask-Detection

install the trained weighed model .pth file from the link
https://rip123-my.sharepoint.com/:u:/g/personal/mubi_rip123_onmicrosoft_com/IQB8t_GWRem2RLgRvbKZrg4BAVpuV0nhYWQjV76eXY-9I6U?e=Up8lnG
file structure
Webcam-Mask-Detection
->  rest of the files from the repo
--> .pth weighed model file.
install python 3.11
open powershell after installing python

**CPU only (no GPU)**
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pillow mediapipe

**CUDA 12.1 (most common modern GPU)**
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy pillow mediapipe

**CUDA 11.8 (older GPU / driver)**
bashpip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pillow mediapipe

**With pinned versions (most stable for Python 3.11)**
bashpip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python==4.10.0.84 numpy==1.26.4 Pillow==10.3.0 mediapipe==0.10.14

from powershell **python ./webcam.py**
