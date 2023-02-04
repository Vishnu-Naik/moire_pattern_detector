# # Classifier
# SVM classifier

# # Hyperparameter search

# ## Grid search
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import numpy as np
from preprocessing.data_preprocess import get_input_features
import logging

logging.getLogger('preprocessing').setLevel(logging.ERROR)



now = datetime.now()
DATE_STRING = now.strftime("%d_%m_%Y_%H_%M_%S")

CUR_DIR = Path.cwd()
DATA_DIR = CUR_DIR / '../data/Train'
POSITIVE_IMG_DIR_PATH = str(Path(DATA_DIR / 'positiveImages').resolve())
NEGATIVE_IMG_DIR_PATH = str(Path(DATA_DIR / 'negativeImages').resolve())

X_train_True = get_input_features(POSITIVE_IMG_DIR_PATH)
Y_train_True = np.ones(len(X_train_True))
X_train_False = get_input_features(NEGATIVE_IMG_DIR_PATH)
Y_train_False = np.zeros(len(X_train_False))

X_train = np.concatenate((X_train_True, X_train_False))
Y_train = np.concatenate((Y_train_True, Y_train_False))


def get_best_hyperparameter(feature_array, target):
    """
    Find the best hyperparameters for a support vector machine (SVC) model.

    :param feature_array: A numpy array of the features to use for training the model.
    :param target: A numpy array of the target values corresponding to the features.

    :returns: A dictionary of the best hyperparameters for the model.
    """
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    print('Performing grid search...')
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(feature_array, target)
    print(f'The best parameters are {grid.best_params_} with a score of {grid.best_score_:.2f}')
    return grid.best_params_


svm_hyperparameters = get_best_hyperparameter(X_train, Y_train)

print('Training SVM classifier...')
svm_classifier = SVC(C=svm_hyperparameters['C'], gamma=svm_hyperparameters['gamma'], verbose=True)

svm_classifier.fit(X_train, Y_train)
print('SVM classifier trained.')
# save the model to disk
CLASSIFIER_FILENAME = f'multi_feature_model_{DATE_STRING}.joblib'
CLASSIFIER_DIR = CUR_DIR / 'saved_models'
CLASSIFIER_PATH = CLASSIFIER_DIR / CLASSIFIER_FILENAME
joblib.dump(svm_classifier, CLASSIFIER_PATH)
print(f'Model saved to {CLASSIFIER_PATH}')
