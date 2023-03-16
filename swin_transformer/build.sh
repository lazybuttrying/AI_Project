# ~/바탕화면/dataset_CVAT/surprise

# Check to CUDA in docker container
# python -c "import torch; torch.cuda.get_device_name(0)"

docker build -t mmdetection docker/
docker run --gpus all --shm-size=16g -it -v /home/agtech-research/바탕화면/dataset_CVAT/surprise/mmdetection/dataset:/mmdetection/dataset mmdetection