# ~/바탕화면/dataset_CVAT/surprise

docker build -t mmdetection docker/
docker run --gpus all --shm-size=8g -it -v /home/agtech-research/바탕화면/dataset_CVAT/surprise/mmdetection/dataset:/mmdetection/dataset mmdetection