# Webcam-Mask-Detection

Install the trained weighed model `.pth` file from the link:

[Download Model](https://rip123-my.sharepoint.com/:u:/g/personal/mubi_rip123_onmicrosoft_com/IQB8t_GWRem2RLgRvbKZrg4BAVpuV0nhYWQjV76eXY-9I6U?e=Up8lnG)

## File Structure
Webcam-Mask-Detection
├─ rest of the files from the repo
└─ .pth weighed model file

## Installation 

### Python 3.11

1. Install Python 3.11.
2. Open PowerShell after installing Python.

### CPU only (no GPU)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pillow mediapipe
```

From PowerShell:
python ./webcam.py
