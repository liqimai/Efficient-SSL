import numpy as np
from gcn.graphconv import ap_approximate


def Model17(adj, alpha, y_train, y_test):
    k = int(np.ceil(4 * alpha))
    prediction, time = ap_approximate(adj, y_train, alpha, k)
    predicted_labels = np.argmax(prediction, axis=1)
    prediction = np.zeros(prediction.shape)
    prediction[np.arange(prediction.shape[0]), predicted_labels] = 1

    test_acc = np.sum(prediction * y_test) / np.sum(y_test)
    test_acc_of_class = np.sum(prediction * y_test, axis=0) / np.sum(y_test, axis=0)
    return test_acc, test_acc_of_class
