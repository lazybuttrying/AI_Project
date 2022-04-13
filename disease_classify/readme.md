### Dataset
- {'노균병': 0, '녹병': 1, '잿빛곰팡이병': 2, '정상': 3, '흰가루병': 4}
- [77, 59, 196, 359, 72]
- total : 763
- All the disease pictures from google search

### Configuration
- model : swin_base_patch4_window7_224
- optimizer : QHAdam
- loss function : FocalLossFlat
- Augmentation : Brightness(), Contrast(), Hue(), Saturation()

### Struggles...
- 노균병 제외하면 모두 정답
  - ~~연한 노균병의 경우 정상으로 오인
  - ~~녹병도 노란색을 띄기에 노균병으로 오인함...
