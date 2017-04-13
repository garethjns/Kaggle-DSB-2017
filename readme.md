# Kaggle Data Science Bowl 2017

[https://www.kaggle.com/c/data-science-bowl-2017](https://www.kaggle.com/c/data-science-bowl-2017)

Final positions: 350/2000 (stage 1), 188/415 (stage 2).

## Aim
 - Predict the chance of a patient developing cancer within the next year using 3D lung CT scans.

# Approach

The data for the competition was slightly different to previous competition such as LUNA16, in that the only labelling of the data contained only an indication of whether or not a patient would go on to develop cancer with the next year. No lung nodule locations were provided (or labels of their expected malignancy). Two main steps are therefore to predict patient outcome:

- Find candidate nodule locations
- Predict malignancy of nodules

My approach focused on the modelling aspects of the task, with the aim of gaining more experience training and using deep neural networks in TensorFlow and Keras. I think significant improvements could be made by taking more care with preprocessing and with categorising of individual nodules after detection.

## Preprocess patient data.
**preprocessing.py**

 - Isolate lungs, remove blood vessels, etc. 
 - Implements modified versions of [Guido Zuidhof's](https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial) kernel and [ArnavJain's](https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing) kernel.

## Preprocessing [LUNA16](https://luna16.grand-challenge.org/) and training [UNET](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)
**UNET.py**
- Preprocess LUNA16 data
- Train UNET model.
- Using the nodule locations and 3D lung data available in the LUNA16 dataset, train UNET to predict candidate nodule locations.  


## Finding candidate nodules
**generateFeatures.py**
- Predict candidate nodule locations using simple image analysis techniques.  

**UNET.py**
- Predict candidate nodule locations based on predictions from trained UNET model.


## Models
**XGBMod.py**
- Manually extract meta features from preprocessing stages and nodule extraction and train a classification tree using XGBoost. 

**tfMod.py**
- Train a 3D convolutional neural network to predict malignancy of candidate nodules. 


## Ensembling
**ensemble.py**
 - Create ensemble of predictions from XGBMod models (Tree and DART) and 3D NN models using UNET and basic nodule predictions.

# To do

- Add UNET training code to UNET.
- Finish adding training and prediction code to tfMod.

