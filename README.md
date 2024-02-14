# 草莓识别
中文 | [英文](./README_EN.md)
## 课题背景
主办方要求参赛人员使用主办方提供的数据训练模型并能精准识别测试数据集中100张图片中的**可采摘**草莓的**形状**和**位置**。
## 特征解释
|  |  |
|--|--|
|  **StrawberryIrremovable**| **StrawberryRemovable** |
|  草莓不可采摘| 草莓可采摘|
|  **StrawberryOther**| **StrawberryFusiform** |
|  其他形状草莓| 纺锤形草莓|
|  **StrawberryCone**| **StrawberryCircular** |
|  球形草莓| 环状草莓|
| **StrawberryWedge**|
| 楔形草莓 |
## 数据处理
主办方提供的数据集分为两部分，一部分是100张**已标注**的草莓图片（即这些图片中的草莓位置有文本标注），另一部分是300张**未标注**的草莓图片。
由于主办方提供的数据集量小，不足以训练深度学习模型，所以我们先将100张图片经过**图片增强**处理，以增加图片数量。
下图是100张已标注草莓图片经过图片增强后的特征分布：
<img src="https://github.com/Fent1/object-detection/assets/43925272/0e182a90-a3a9-4cc6-a8c7-4e6a7581fc5e" alt="image" width="300" height="auto">
通过上图草莓图片特征分布可以看出特征的分布非常分散，特别是**草莓形状**的特征（第3到8列的特征）数量相对**草莓是否可采摘**（第1,2列特征）非常稀少。特征的不平衡会导致训练结果精度很低，所以我们决定采用多种方式来解决数据特征不平衡的影响。

1. 利用300张未标注草莓图片手动增加**草莓形状**特征的数量，处理后的草莓特征分布如下：

<img src="https://github.com/Fent1/object-detection/assets/43925272/32ac445a-5cb3-4f51-a0d2-a19867ecc27a" alt="image" width="300" height="auto">

2. 在训练模型时为草莓形状的特征赋予更高的权重，以抵消**草莓形状**特征数量少的影响。

## 模型选择
该课题更符合目标检测模型，因为目标检测模型能很好地检测到图片中目标的位置。而在众多目标检测模型中，我们选择YOLOv5框架来训练模型，原因有：

 1. YOLO系列模型是高度成熟的目标检测模型，它相较于传统CNN模型训练速度快了1000倍，而且使用方便。
 2. YOLO系列模型已经更新到YOLOv8，但是我们选择使用YOLOv5是因为YOLOv8的集成做得没有YOLOv5稳定，训练模型时出现很多报错，所以我们退而求其次选择更为稳定更成熟的YOLOv5框架，他的特征提取能力是仅次于YOLOv8的。
![image](https://github.com/Fent1/object-detection/assets/43925272/4d902c71-c8c7-41e4-969b-9a5a9ce03112)

## 模型表现
本次训练后模型的mAP_0.5为0.3402，虽然分数很低，但本次比赛中最高分数也仅仅是0.4301，显然是因为数据集中草莓特征分布不平衡的影响导致的。

<img src="https://github.com/Fent1/object-detection/assets/43925272/96f23991-a557-490e-8cfd-2c7f04acb2d5" alt="image" width="300" height="auto">

## 未来解决思路
使用YOLOv8的segmentation模型，segmentation模型需要的目标物体标准方式相较于普通模型有区别。普通模型使用四边形标注法，即下图所示：
![image](https://github.com/Fent1/object-detection/assets/43925272/478aa519-44f0-41d9-ac3a-f5c8fedbe87a)

而segmentation模型使用多边形标注法，即下图所示：
![image](https://github.com/Fent1/object-detection/assets/43925272/f60be82d-553a-435d-8571-710d37c14bfe)

这样的优点就是能更好地提取目标物体的形状特征，这样可以更大程度地提高模型分数。

