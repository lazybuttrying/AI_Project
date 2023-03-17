
- Main Repository: https://github.com/open-mmlab/mmdetection

- Dataset
  - All of the image date have to be stored in folder 'dataset/image/'
  - The content of .txt file is the only image file name without image file type
  - Label data is stored in label.xml. Points of bounding box is loaded before training

- Dockerfile
  - To run on the RTX 3080, Each version of Pytorch, CUDA and CUDNN is changed
  
```
2023-03-17 08:16:35,030 - mmdet - INFO - workflow: [('train', 1)], max: 12 epochs
2023-03-17 08:16:35,030 - mmdet - INFO - Checkpoints will be saved to /mmdetection/tutorial_exps by HardDiskBackend.
Traceback (most recent call last):
  File "/mmdetection/tutorial-strawberry.py", line 268, in <module>
    train_detector(model, datasets, cfg, distributed=False, validate=True)
  File "/mmdetection/mmdet/apis/train.py", line 246, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/opt/conda/lib/python3.10/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/mmcv/runner/epoch_based_runner.py", line 50, in train
    self.run_iter(data_batch, train_mode=True, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/mmcv/runner/epoch_based_runner.py", line 29, in run_iter
    outputs = self.model.train_step(data_batch, self.optimizer,
  File "/opt/conda/lib/python3.10/site-packages/mmcv/parallel/data_parallel.py", line 67, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/mmdetection/mmdet/models/detectors/base.py", line 248, in train_step
    losses = self(**data)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/mmcv/runner/fp16_utils.py", line 98, in new_func
    return old_func(*args, **kwargs)
  File "/mmdetection/mmdet/models/detectors/base.py", line 172, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/mmdetection/mmdet/models/detectors/two_stage.py", line 127, in forward_train
    x = self.extract_feat(img)
  File "/mmdetection/mmdet/models/detectors/two_stage.py", line 67, in extract_feat
    x = self.backbone(img)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mmdetection/mmdet/models/backbones/swin.py", line 763, in forward
    x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mmdetection/mmdet/models/backbones/swin.py", line 457, in forward
    x = block(x, hw_shape)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mmdetection/mmdet/models/backbones/swin.py", line 376, in forward
    x = _inner_forward(x)
  File "/mmdetection/mmdet/models/backbones/swin.py", line 363, in _inner_forward
    x = self.attn(x, hw_shape)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mmdetection/mmdet/models/backbones/swin.py", line 231, in forward
    attn_windows = self.w_msa(query_windows, mask=attn_mask)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mmdetection/mmdet/models/backbones/swin.py", line 90, in forward
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 842.00 MiB (GPU 0; 9.77 GiB total capacity; 6.92 GiB already allocated; 502.81 MiB free; 7.21 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```