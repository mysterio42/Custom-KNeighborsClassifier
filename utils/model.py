from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import string
import glob
import os
import numpy as np
import random
from utils.plot import plot_cm
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """

    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def generate_model_name(size=5):
    """
    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def load_model():
    """
    :return: load latest modified weight model
    """
    path = latest_modified_weight()

    with open(path, 'rb') as f:
        return joblib.load(filename=f)

def find_optimal_k(features, labels):
    """

    :param features: 'x'', 'y',
    :param labels: cluster_key
    :return:
    """
    scores = []
    k_range = range(1, 50)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, features, labels, cv=10)
        scores.append(score.mean())
    opt_k = np.argmax(scores) + 1  # because index starts from zero !
    opt_score = np.amax(scores)
    return k_range, scores, opt_k, opt_score


def train_model(features, labels,opt_k):
    """
      run train_test_split method with optimal K
      :param features: 'x'', 'y'
      :param labels: cluster_key
      :return:
      """

    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3)

    model = KNeighborsClassifier(n_neighbors=opt_k)
    model.fit(feature_train, label_train)

    preds = model.predict(feature_test)

    cm = confusion_matrix(label_test, preds)

    plot_cm(cm)

    print(accuracy_score(label_test, preds))

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        model_name = WEIGHTS_DIR + 'knn-' + generate_model_name(5) + '.pkl'
        with open(model_name, 'wb') as f:
            joblib.dump(value=model, filename=f, compress=3)
            print(f'Model saved at {model_name}')
    return model

