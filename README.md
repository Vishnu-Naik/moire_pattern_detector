# Moire Pattern Detection
## Introduction
This repository contains a Python code for detecting moire patterns in images. The code performs data preprocessing, which is a crucial step in the moire pattern detection process, where multiple features are extracted from the input images. These features are then injected into a Support Vector Machine (SVM) classifier to classify the images as containing a moire pattern or not. The SVM hyperparameters are obtained through a grid search, ensuring that the best hyperparameters are used for the classification task.

## Requirements
The code has been tested with the following dependencies:

- Python 3.10 or later
- NumPy
- OpenCV
- Scikit-learn
- Scikit-image

## Data Preprocessing
The first step in the moire pattern detection process is data preprocessing. This involves extracting various features from the input images that can be used to classify the images as containing a moire pattern or not. The code in this repository implements multiple feature extraction techniques, including:

- Color Histogram
- Distance Histogram
- Local Binary Pattern (LBP)

## Support Vector Machine (SVM) Classifier
Once the features have been extracted, they are fed into a Support Vector Machine (SVM) classifier. The SVM is trained on a labeled dataset to learn the relationship between the extracted features and the presence or absence of a moire pattern in the image. The SVM hyperparameters are obtained through a grid search, ensuring that the best hyperparameters are used for the classification task. The trained SVM can then be used to classify new images as containing a moire pattern or not.

# Running the Code
The code is organized into the following modules:

- `preprocessing`: Contains the functions for data preprocessing and feature extraction.
- `classifier.py`: Contains the functions for training and saving the SVM classifier to the disk.
- `inference.py`: This module contains the functions for loading the trained SVM classifier and using it to classify new images.
To run the code, simply execute the following commands in the terminal:

To train the classifier:
```python classifier.py```

To classify new images:
```python inference.py```

*Note:* The code assumes that the training and test images are stored in the `data` folder. 
The `data` folder should contain two subfolders, `Train` and `Test`, which contain the training and test images, respectively. 
The `train` folder should contain two subfolders, `positiveImages` and `negativeImages`, which contain the images containing a moire pattern and images without a moire pattern, respectively. 
The `test` folder should contain the images to be classified. Make changes to the code as necessary to suit your use case.
The data folder should reside in the same level as `src`. Data is not included in this repository because the data was collected from kaggle from this [link](https://www.kaggle.com/datasets/dataclusterlabs/real-world-and-moire-pattern-classification?resource=download)
Download the data and place it in the data folder at your discretion.

## Conclusion
This repository provides a complete implementation of moire pattern detection in Python. The code is well-commented and organized, making it easy to understand and modify for your own use case. If you have any questions or feedback, feel free to reach out.