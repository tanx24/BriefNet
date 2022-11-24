# Bridging Feature Complementarity Gap between Encoder and Decoder for Salient Object Detection
by Zhenshan Tan, Xiaodong Gu

## Introduction
Deep convolutional neural networks have pushed the performances of salient object detection to the new state-of-the-art. However, most of the existing methods mainly focus on the calibration of the encoder or the decoder while ignore the complementarity between them, which may lose some vital saliency areas or produce noises. Moreover, the research on the saliency structure-level loss function still remains scarce. To address the above issues, we propose a bridging feature complementarity gap network (BriefNet) which consists of a common encoder-decoder structure, an interactive complementarity module (ICM), an encoder clustering module (ECM) and a feature aggregation module (FAM) optimized by a novel hybrid loss. Specifically, ICM and ECM are embedded after the encoder-decoder structure to recalibrate the features of encoder and decoder. Firstly, ICM includes a feature supplement operation (FSO) and a feature generality extraction (FGE). FSO makes up the lost features of the decoder. FGE aims at exploring the generality between the encoder and the decoder, which helps to constrain the saliency. Secondly, ECM is combined to the decoder features to further enhance the connectivity of the encoder features by considering the local clustering features and the global connective features of the encoder, aiming at aggregating the relevancy semantic information. Thirdly, ICM and ECM are aggregated by FAM to guide the network to bridge the complementarity gap between the encoder and the decoder. Finally, to further recalibrate the fuzzy structure features, we propose a structure recalibration loss (SRC) as the supplementary structure-level loss to the pixel-level binary cross entropy loss (BCE) and the map-level Dice loss. Experimental results on five widely used datasets show that the proposed BriefNet achieves consistently superior performance under various evaluation metrics. In addition, we also put forward a simplified BriefNet, which also achieves competitive results with only increasing a few parameters (10M parameters) compared with the baseline.

## Note
The codes are divided into two parts, including the complete version (140M) and the simple version (110M). The complete version is labelled as **BriefNet** and simple version is labelled as **DFCN**.

## Training
- If you want to train the model by yourself, please download the [pretrained model](https://drive.google.com/file/d/1oJ7-YDGJQAc_H8rS4KLLcpzzOdiALXSk/view?usp=sharing) into BriefNet folder

```
cd BriefNet/
python3 train_BriefNet.py  or python3 train_DFCN.py
```

## Testing
```
cd BriefNet/
python3 test_BriefNet.py or python3 test_DFCN.py 
```

## Saliency maps & Trained model
- saliency maps: [Google (BriefNet)](https://drive.google.com/file/d/187dAU_MDboQrYvrxy8LgHpMFhrgaq4yl/view?usp=sharing) | [Google (DFCN)](https://drive.google.com/file/d/1pQdvnisIqvavDeZfvcBKEslOF0I1RwRG/view?usp=sharing)

- trained model: [Google (BriefNet)](https://drive.google.com/file/d/1oOzYhM58rcOj1aYqCpEuxsRHwM6Q7xD7/view?usp=sharing) | [Google (DFCN)](https://drive.google.com/file/d/1vl7Wg3rrQDfQ4G-BuXnW7Ph_m3KAkafM/view?usp=sharing)


