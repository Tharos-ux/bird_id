# Birdcall identification

## Context :
Do you hear the birds chirping outside your window? 
Over 10,000 bird species occur in the world, and they can be found in 
nearly every environment, from untouched rainforests to suburbs and even cities. 
Birds play an essential role in nature. They are high up in the food chain and 
integrate changes occurring at lower levels. As such, birds are excellent indicators of deteriorating habitat quality and environmental pollution. However, it is often easier to hear birds than see them. With proper sound detection and classification, researchers could automatically intuit factors about an area’s quality of life-based on a changing bird population. it can be downloaded from the Kaggle website here.
The challenge is to build  machine learning algorithm(s) to predict the bird species from audio records. 
Deeplearning is only one of the possible solution.


## Method and steps :
1. Data pre-processing
- Import audios and convert them into spectrograms (choose parameters)
- Data augmentation to limit overfitting
...

2. Neural Network construction
- Spectrogram are analysed as images
- Our program will have to associate patterns in those images to a specie
...


## Usage :

Creates a model and saves it, execute predictions
```bash
python .\main.py -p -o
```

Load a pre-saved model and execute predictions
```bash
python .\main.py -p -m .\models\model_04-11-2022_16-46-07
```