# Human-machine Collaborative (H-M-C) DL Development Code
Here, we show the code proposed by this study for the comparison studies (classic model training and pure machine annotation)

## CODE overview
+ CODE 1: code for classic model training in the model training comparison test
+ CODE 2: code for pure machine annotation in the annotation comparison test

### Dataset
+ Drusen dataset
+ DME dataset
+ Ocular pathology dataset
+ Retinal haemorrhage dataset
+ RDR dataset
+ Spine X-ray dataset
+ Gastrointestinal endoscopy (binary classification) dataset
+ Slit-lamp dataset
+ OCT (multiclass classification) dataset
+ Gastrointestinal endoscopy (multiclass classification) dataset
+ Blood cell dataset

### CODE 1
In the model training comparison test, we utilised the training datasets with ground truth labels to develop classic convolutional neural networks with ResNet-50, Inception-V3, Inception-ResNet-V2, and EfficientNet-B4 network architectures

#### Pre-train
+ Python 3.8
+ keras 2.2.4
+ pip install -r requirements.txt

#### Project structure
```
-CODE 1
  └ requirements.txt
  └ Readme_for_model_download.docx
  └ Project 1 (ex: Blood cell, DME, RDR, etc.)
    └ project
      └ validation_generator.py
      └ train.py
      └ predict.py
  └ Project 2
  └ ...
```

#### Training and testing
```
cd /CODE1/{each dir}/project

python train.py # for train
or
python predict.py  # for test
```
### CODE 2
For the pure machine method on the classification datasets, a series of unsupervised clustering algorithms were applied to cluster the images based on the representation vectors generated by a ResNet-50 network or a denoising autoencoder

#### Pre-train
+ Python 3.6
+ pip install -r requirements.txt

#### Project structure
```
-CODE 2
  └  requirements.txt
  └  autoencoder.py
  └  resnet50.py 
```

#### Clustering for pure machine annotation
```
cd /CODE2/

python autoencoder.py # training denoising autoencoder, encoding images using denoising autoencoder and then clustering for pure machine annotation
or
python resnet50.py  # encoding using resnet50 and then clustering for pure machine annotation
```

### License
These codes are provided for review only. For researchers to get access to the codes, you need sign the [license](LicenseforHMC.pdf) and send it to: linht5@mail.sysu.edu.cn