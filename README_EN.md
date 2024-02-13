# Strawberry Recognition
[Chinese](./README.md) | English
## Background
The organizers require participants to use the provided data to train models and accurately identify the **shape** and **position** of **pickable** strawberries in 100 images in the test dataset.
## Data Processing
The dataset provided by the organizers is divided into two parts: 100 **annotated** strawberry images (i.e., the positions of strawberries in these images are labeled with text) and 300 **unannotated** strawberry images.
Since the dataset provided by the organizers is small and insufficient to train deep learning models, we first augment the 100 images to increase the number of images.
The following figure shows the feature distribution of the 100 annotated strawberry images after image augmentation:

<img src="https://github.com/Fent1/object-detection/assets/43925272/0e182a90-a3a9-4cc6-a8c7-4e6a7581fc5e" alt="image" width="300" height="auto">

From the feature distribution of the strawberry images shown above, it can be seen that the distribution of features is very scattered, especially the **shape** features of strawberries (features in columns 3 to 8) are relatively sparse compared to **whether strawberries are pickable** (features in columns 1 and 2). The imbalance of features will result in low accuracy of the training results, so we have decided to use multiple methods to address the impact of data feature imbalance.

1. Manually increase the number of **shape** features of strawberries using 300 unannotated strawberry images. The distribution of processed strawberry features is as follows:
   
<img src="https://github.com/Fent1/object-detection/assets/43925272/32ac445a-5cb3-4f51-a0d2-a19867ecc27a" alt="image" width="300" height="auto">

2. Assign higher weights to the **shape** features of strawberries during model training to offset the influence of the small number of **shape** features.

## Model Selection
This project is more suitable for object detection models because object detection models can accurately detect the positions of objects in images. Among many object detection models, we choose the YOLOv5 framework to train the model for the following reasons:

1. The YOLO series models are highly mature object detection models, which are 1000 times faster in training speed compared to traditional CNN models and easy to use.
2. Although the YOLO series models have been updated to YOLOv8, we choose to use YOLOv5 because the integration of YOLOv8 is not as stable as YOLOv5. When training the model, many errors occurred with YOLOv8, so we choose the more stable and mature YOLOv5 framework, which has feature extraction capabilities second only to YOLOv8.

![image](https://github.com/Fent1/object-detection/assets/43925272/4d902c71-c8c7-41e4-969b-9a5a9ce03112)

## Model Performance
The mAP_0.5 of the model trained this time is 0.3402. Although the score is very low, the highest score in this competition is only 0.4301, obviously due to the imbalance of strawberry feature distribution in the dataset.

<img src="https://github.com/Fent1/object-detection/assets/43925272/96f23991-a557-490e-8cfd-2c7f04acb2d5" alt="image" width="300" height="auto">

## Future Solutions
Using the segmentation model of YOLOv8, the standard way of marking objects required by the segmentation model is different from that of ordinary models. Ordinary models use the quadrilateral labeling method, as shown below:

![image](https://github.com/Fent1/object-detection/assets/43925272/478aa519-44f0-41d9-ac3a-f5c8fedbe87a)

While segmentation models use the polygon labeling method, as shown below:

![image](https://github.com/Fent1/object-detection/assets/43925272/f60be82d-553a-435d-8571-710d37c14bfe)

The advantage of this method is that it can better extract the shape features of the target object, which can greatly increase the model score.
