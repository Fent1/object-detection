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
## 图片增强
对原图做 随机改变亮暗、对比度和颜色等 数据增强：

![image](https://github.com/Fent1/object-detection/assets/43925272/8557a87e-83ca-44a8-a9a0-407465cfac69)

随机缩放：

![image](https://github.com/Fent1/object-detection/assets/43925272/2a97a564-e9c6-4565-9d75-1ca36568d728)

随机翻转：

![image](https://github.com/Fent1/object-detection/assets/43925272/0ffb88c1-abe7-4421-91aa-5274b382b85f)



下图是100张已标注草莓图片经过图片增强后的特征分布：

<img src="https://github.com/Fent1/object-detection/assets/43925272/0e182a90-a3a9-4cc6-a8c7-4e6a7581fc5e" alt="image" width="300" height="auto">

通过上图草莓图片特征分布可以看出特征的分布非常分散，特别是**草莓形状**的特征（第3到8列的特征）数量相对**草莓是否可采摘**（第1,2列特征）非常稀少。特征的不平衡会导致训练结果精度很低，所以我们决定采用多种方式来解决数据特征不平衡的影响。

1. 利用300张未标注草莓图片增加**草莓形状**特征的数量：
   1) 将100张已标注的图片去掉**草莓不可采摘**和**草莓可采摘**的标注，将处理后的100张图片进行**迁移训练**：
   
   <img src="https://github.com/Fent1/object-detection/assets/43925272/2ffaf11f-4219-4f1b-b37f-6d3b72be3d2e" alt="image" width="300" height="auto">


   2) 将迁移训练后的模型用来预测300张未标注的草莓图片，此时的300张草莓图片均已被标注；
   3) 最后将300张图片与原100张图片（含**草莓不可采摘**和**草莓可采摘**的标注）合并，处理后的草莓特征分布如下：

<img src="https://github.com/Fent1/object-detection/assets/43925272/32ac445a-5cb3-4f51-a0d2-a19867ecc27a" alt="image" width="300" height="auto">

值得注意的是，迁移训练后预测得到的300张被标注图片的准确率与最终模型的分数非常相近，但是迁移训练本身并不是影响模型最终分数的因素。

2. 在训练模型时为草莓形状的特征赋予更高的权重，以抵消**草莓形状**特征数量少的影响。

## 什么是迁移训练?
迁移学习是根据新数据快速重新训练模型的有效方法，而无需重新训练整个网络。在转移学习中，部分初始权重被**冻结**在原位，其余权重用于计算损失，并由优化器进行更新。这比普通训练所需的资源更少，训练时间也更短，但也可能导致最终训练精度的降低。
本次使用迁移训练的目的就是在**尽可能减少后续模型训练精度影响**的情况下**确保数据集特征的一致性**

## 冻结骨干层
与train.py匹配的所有图层 freeze 在训练开始前，train.py列表中的梯度将被设置为零，从而被冻结。

            # Freeze
            freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
            for k, v in model.named_parameters():
              v.requires_grad = True  # train all layers
              if any(x in k for x in freeze):
                  print(f'freezing {k}')
                  v.requires_grad = False
查看模块名称列表：

            for k, v in model.named_parameters():
              print(k)
            
            """Output:
            model.0.conv.conv.weight
            model.0.conv.bn.weight
            model.0.conv.bn.bias
            model.1.conv.weight
            model.1.bn.weight
            model.1.bn.bias
            model.2.cv1.conv.weight
            model.2.cv1.bn.weight
            ...
            model.23.m.0.cv2.bn.weight
            model.23.m.0.cv2.bn.bias
            model.24.m.0.weight
            model.24.m.0.bias
            model.24.m.1.weight
            model.24.m.1.bias
            model.24.m.2.weight
            model.24.m.2.bias
            """
            
骨干层的主要作用是在训练模型时提取训练集中的特征，通过学习目标的特征进而提高自己对特定目标的识别能力。
纵观模型结构，我们可以看到模型的主干是 0-9 层：

            # YOLOv5 v6.0 backbone
            backbone:
            # [from, number, module, args]
            - [-1, 1, Conv, [64, 6, 2, 2]]  # 0-P1/2
            - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
            - [-1, 3, C3, [128]]
            - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
            - [-1, 6, C3, [256]]
            - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
            - [-1, 9, C3, [512]]
            - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
            - [-1, 3, C3, [1024]]
            - [-1, 1, SPPF, [1024, 5]]  # 9
            
因此，我们可以将冻结列表定义为包含所有名称中包含 "model.0.- 模型.9. "的模块：

            python train.py --freeze 10
            
由于迁移训练选择的模型一般具有泛化的提取特征能力，冻结骨干架构可以：
  1) 保留预训练模型在源任务上学到的特征提取能力；
  2) 由于迁移训练时的100张已标注图片被我们去掉两个特征，我们不希望模型学习到这个数据集的特征，会影响最终模型对**草莓不可采摘**和**草莓可采摘**特征的判断,
     所以冻结骨干层还能减少后续模型训练受到影响；
  3）由于骨干层被冻结，模型不具备学习新特征的能力，后续的模型训练也不存在过拟合的风险。

<img src="https://github.com/Fent1/object-detection/assets/43925272/71d985f7-dbb8-480b-b85f-c47f171e7bea" alt="image" width="600" height="auto">

冻结骨干层与默认模型相比分数并没有减少很多

## 模型选择
该课题更符合目标检测模型，因为目标检测模型能很好地检测到图片中目标的位置。而在众多目标检测模型中，我们选择YOLOv5框架来训练模型，原因有：

 1. YOLO系列模型是高度成熟的目标检测模型，它相较于传统CNN模型训练速度快了1000倍，而且使用方便。
 2. YOLO系列模型已经更新到YOLOv8，但是我们选择使用YOLOv5是因为YOLOv8的集成做得没有YOLOv5稳定，训练模型时出现很多报错，所以我们退而求其次选择更为稳定更成熟的YOLOv5框架，他的特征提取能力是仅次于YOLOv8的。

<img src="https://github.com/Fent1/object-detection/assets/43925272/4d902c71-c8c7-41e4-969b-9a5a9ce03112" alt="image" width="600" height="auto">

## 模型表现
本次训练后模型的mAP_0.5为0.3402，虽然分数很低，但本次比赛中最高分数也仅仅是0.4301，显然是因为数据集中草莓特征分布不平衡的影响导致的。

<img src="https://github.com/Fent1/object-detection/assets/43925272/96f23991-a557-490e-8cfd-2c7f04acb2d5" alt="image" width="300" height="auto">

## 模型预测
根据预测集的结果可以判断，模型能够辨别大多数的**草莓不可采摘**和**草莓可采摘**特征，这是由于这两个特征在训练集中数量庞大。
而模型能够辨别出少量的**球状草莓**特征，也同样是因为该特征相比于其他草莓形状特征的数量更多。

![image](https://github.com/Fent1/object-detection/assets/43925272/d27745f1-168f-4361-b324-6e8d87ae8320)


## 未来解决思路
使用YOLOv8的segmentation模型，segmentation模型需要的目标物体标准方式相较于普通模型有区别。普通模型使用四边形标注法，即下图所示：
![image](https://github.com/Fent1/object-detection/assets/43925272/478aa519-44f0-41d9-ac3a-f5c8fedbe87a)

而segmentation模型使用多边形标注法，即下图所示：
![image](https://github.com/Fent1/object-detection/assets/43925272/f60be82d-553a-435d-8571-710d37c14bfe)

这样的优点就是能更好地提取目标物体的形状特征，从而更大程度地提高模型分数。

## 应用场景
火山引擎中的**图像技术**业务用到了目标检测技术。
对于**图像技术**板块来说，**图像内容检索**功能与**个性内容推荐**就是目标检测的应用场景，图像内容检索能根据识别的目标打上标签，

![image](https://github.com/Fent1/object-detection/assets/43925272/ddf3ef29-6dea-48de-b800-64cbca6e1c71)


而**个性内容推荐**也会根据识别图像中的物体进行分类，然后再进行个性化的推荐。

![image](https://github.com/Fent1/object-detection/assets/43925272/bb79303f-70c4-43a4-a8ca-2c106268a4d8)


本次草莓分类的模型训练任务虽然分数不高，但是在分类草莓可采摘与否的任务上有显著的优势，说明YOLO对于不同物体的识别分类
是非常有效的。而对于火山引擎的图像业务来说，存在相似物体的分类情况是非常少的，所以YOLO是一个火山引擎所需要的图像识别模型。

