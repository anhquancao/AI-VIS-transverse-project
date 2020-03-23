# Real-time Arbitrary Style Transfer for Selfie image with Facial Structure Preservation

Our implementation is based on the implementation of AdaIN from https://github.com/naoto0804/pytorch-AdaIN.
The BiSeNet implementation: https://github.com/zllrunning/face-parsing.PyTorch



![Results](all_results.png)

## Requirements
Please install requirements by `pip install -r requirements.txt`

- Python 3.5+
- PyTorch 0.4+
- TorchVision
- Pillow

(optional, for training)
- tqdm
- TensorboardX

## Usage

### Download models
Download https://drive.google.com/drive/folders/1xjHf_iUDOU07PCbDhcdVSVgr0quQx8GF?usp=sharing

### Test
Notebooks:
- test_depth_AdaIN.ipynb: Depth AdaIN
- test_depth_face_segmention_AdaIN.ipynb: Face Segmentation and Depth AdaIN
- test_face_segmentaion_AdaIN.ipynb: Face Segmentation AdaIN

### Train

Notebooks:
- train_depth_AdaIN.ipynb: Train depth AdaIN
- train_depth_and_face_segment_AdaIN.ipynb: Train AdaIN with both Depth and Face Segmentation
- train_face_segmentation_AdaIN.ipynb: Train Face Segmentation AdaIN
- train_depth_estimation.ipynb: Train depth estimation network
