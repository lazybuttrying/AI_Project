
- Main Repository: https://github.com/open-mmlab/mmdetection

- Dataset
  - All of the image date have to be stored in folder 'dataset/image/'
  - The content of .txt file is the only image file name without image file type
  - Label data is stored in label.xml. Points of bounding box is loaded before training

- Dockerfile
  - To run on the RTX 3080, Each version of Pytorch, CUDA and CUDNN is changed

- config
  - configs/strawberry/fast_rcnn_with_swin.py

- According to type of Computer Vison ...
  - To do Object Detction, use FastRCNN (current repository)
  - To do Instance Segmentation, use MaskRCNN


- why...
  - no result on every image...
  - validation is gone...
