# Bridging Feature Complementarity Gap between Encoder and Decoder for Salient Object Detection

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
