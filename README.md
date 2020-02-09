# Cascaded Context Dependency: An Extremely Lightweight Module for Deep Convolutional Neural Networks
Xu Ma, Zhinan Qiao, Jingda Guo, Sihai Tang, Qi Chen, Qing Yang, Song Fu<br>
(Submitted to ICIP2020)
<br><br>


## Implementation
In this repository, all the models are implemented by [pytorch](https://pytorch.org/).<br><br>

We use the standard data augmentation strategies with [ResNet](https://github.com/pytorch/examples/blob/master/imagenet/main.py).<br><br>

To reproduce our CCD module work, please refer [Usage.md](Usage.md).

## Trained Models

:blush: `All trained models and training log files are submitted to Google Drive.`

:blush: `We provide corresponding links in the "download"  column.`

<br>
<br>
Table 1:  Comparison results of single-crop classification accuracy (%) and complexity on the ImageNet validation set.  The best two performances are marked in **bold**.

| Model | top-1 acc. |top-5 acc. |FLOPs(G)|Parameters(M)|Download|
| --- | --- |--- |--- |--- |---|
| ResNet50 | 75.8974 |92.7224|4.122|25.557|<a href="https://drive.google.com/open?id=1DMHhk99fG8rNZjE2wPh8VWZ5qIBOaYOf">model</a> <a href="https://drive.google.com/open?id=1KOM5BzyxQLZl2Aa5KIVOh6HmE7eQvsKa">log</a>|
| SE-ResNet50 |77.2877  |**93.6478**|4.130|28.088|<a href="https://drive.google.com/open?id=1lOXZv0IskrLLbm_z7JqonR6KaQ7lRpKP">model</a> <a href="https://drive.google.com/open?id=1gl43ufL2Pvum-dZy8B4yAnnV3bl1BSi2">log</a>|
| GE-ResNet50 |76.2357  |92.9847|**4.127**|**25.557**|<a href="https://drive.google.com/open?id=1N-UVJhZDkoHnxzhE0p_VRsCgGDExi0iA">model</a> <a href="https://drive.google.com/open?id=1KcPMHcDfcgu87TAHqy3ovNN29pIZdkPi">log</a> |
| CBAM-ResNet50 | 77.2840 |93.6005|4.139|28.090|<a href="https://drive.google.com/open?id=1brCXiQ0LNbqVejQMrY0eVmcwZSGhYFN3">model</a> <a href="https://drive.google.com/open?id=1MVBSKLSu9lyxNKrxH4WoA4fHsE86y45K">log</a> |
| SK-ResNet50 | **77.3657** |93.5256|4.187|26.154|<a href="https://drive.google.com/open?id=1jwQ-us0G0LSesHGZwmDgjL1W5O7Ekyu6">model</a> <a href="https://drive.google.com/open?id=1DGMM9c1Vfo_YniYTUeL-jfAmsQuwhJYX">log</a> |
| GC-ResNet50 |74.8966  |92.2812|4.130|28.105|<a href="https://drive.google.com/open?id=1GGe9UzVFjMpoRkQVf3td5BrLeb1ZfwVM">model</a> <a href="https://drive.google.com/open?id=1iE8m0MgK8Ek7ui5UxF8s0w8tZ8XG8dyN">log</a> |
| CCD-ResNet50 (ours) | **77.3137** |**93.6489**|**4.122**|**25.560**|<a href="https://drive.google.com/open?id=1mHqmrkrWudCk-3DCXL8XWbIjBznOjDLh">model</a>  <a href="https://drive.google.com/open?id=1ZLNEEXAdCUrILmkjZMG_-nYjKYGhS8De">log</a> |




<br>
<br>
Table 2: Detection performances (%) with different backbones on the MS-COCO validation dataset. We employ two state-of-the-art detectors: RetinaNet and Cascade R-CNN  in our detection experiments.

| Detector | Backbone | AP(50:95) | AP(50) | AP(75) | AP(s)|AP(m)|AP(l)|Download
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Retina|ResNet50|36.2|55.9|38.5|19.4|39.8|48.3|[model](https://drive.google.com/open?id=1imZvUrwg6Vy6TFRLAsL62FsF-DyizZXR) [log](https://drive.google.com/open?id=14rRmHai_9ghL5oC-1DTTiLrt4w_HY0Yl)
|Retina|SE-ResNet50|37.4|57.8|39.8|20.6|40.8|50.3|[model](https://drive.google.com/open?id=1ivzPfC_JhpO7DPs6vzlHGxkZBf2sC60p) [log](https://drive.google.com/open?id=1mKctgPjf9QbEXTeSm_-J_kqeiVNGuMT7)
|Retina|CCD-ResNet50|**37.8**|**58.5**|**40.1**|**21.6**|**41.5**|**50.9**|[model](https://drive.google.com/open?id=1StYpULhwgCwG_ZacBR1bRFqbgt6FRHZr) [log](https://drive.google.com/open?id=1ADWdGj2NcuiK2SCExfWKM8ovypBC68FL)
Cascade R-CNN|ResNet50|40.6|58.9|44.2|22.4|43.7|**54.7**|[model](https://drive.google.com/open?id=1jGUT2KsFggLSJMkH0cgJUJV_p_cSM-7f) [log](https://drive.google.com/open?id=13g-4XlMlySVUJyrvWeU5FVCA--cojaCk)
Cascade R-CNN|GC-ResNet50|41.1|59.7|44.6|**23.6**|44.1|54.3|[model](https://drive.google.com/open?id=19cv3TReITDMJuvmAleGzzt3H39iq3pYl) [log](https://drive.google.com/open?id=1uCcKukd4HKtxIc1uUfKydd-_NIPnj9_i)
Cascade R-CNN|CCD-ResNet50|**42.5**|**61.1**|**46.4**|24.7|**45.9**|56.5|[model](https://drive.google.com/open?id=1655frDSIzUpxjOD4Bt2-l6w0D5DBo2Yn) [log](https://drive.google.com/open?id=1655frDSIzUpxjOD4Bt2-l6w0D5DBo2Yn)
