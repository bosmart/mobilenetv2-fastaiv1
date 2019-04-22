# mobilenetv2-fastaiv1
MobileNetV2 for fastaiV1 (pretrained and with [optional] batchnorm fusion) 

Usage (see the Jupyter Notebook for details):

```python
from mbnetv2 import mbnetv2, _mbnetv2_split
learn = cnn_learner(data, mbnetv2, cut=-1, split_on=_mbnetv2_split, metrics=error_rate)
learn.fit_one_cycle(4)
```

Sources:
- https://github.com/tonylins/pytorch-mobilenet-v2
- https://github.com/MIPT-Oulu/pytorch_bn_fusion
