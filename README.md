# Deep_Doc_Classifier

## Requirements:
- tensorflow  
- keras  
- pytorch  
- torchvision  
- pandas  
- matplotlib  
- numpy  
- PIL  
- seaborn  
  
## Files Description:
### data_modification.py
The file data_modification changes the dataset images from a single chanel to 3 channel. This is done as an input to alex net. Also the image size is reduced to 227x227 inorder to reduce the computational overhead.

### config.py
The file config is used to initialize hyperparameters along with other parameters, such as number of epochs, batch-size, and the resize value for the image.

### train_test.py
The train_test file is used to train the model on the new dataset.

### DeepDoc_AlexNet.ipynb
This is the main execution file for the deep doc classifier using the pretrained alex net (pretrained on Imagenet), for classifying documents. There are total of 10 classes.
