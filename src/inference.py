import numpy as np
from preprocessing.data_preprocess import get_input_features
from pathlib import Path
import logging
import joblib

logging.getLogger('preprocessing').setLevel(logging.ERROR)

CUR_DIR = Path.cwd()
DATA_DIR = CUR_DIR / '../data'
CLASSIFIER_DIR = CUR_DIR / 'saved_models'
CLASSIFIER_FILENAME = 'multi_feature_model_04_02_2023_17_37_55.joblib'
CLASSIFIER_PATH = CLASSIFIER_DIR / CLASSIFIER_FILENAME
TEST_IMG_DIR_PATH = Path(DATA_DIR / 'test/negativeImages').resolve()

print('Preparing test data...')
X_test = get_input_features(str(TEST_IMG_DIR_PATH))
# Change below line to 1 if you want to test on positive images
Y_test = np.zeros(len(X_test))
print('Test data prepared.')
print(f'Loading model from {CLASSIFIER_PATH}...')
loaded_model = joblib.load(CLASSIFIER_PATH)
print('Model loaded.')
print('Evaluating model...')
result = loaded_model.score(X_test, Y_test)
print('*' * 100)
print(f'{" " * 50} Results')
print('-' * 100)
print(f'Performance on test set: {result*100:.2f}%')
print('-' * 100)
pred = loaded_model.predict(X_test[2:3])
print(f'Prediction on one test image: {("" if pred[0] == 1 else "No ") + "Moire Pattern Detected"}')
print('*' * 100)
