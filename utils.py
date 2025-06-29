import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV

def load_and_visualize_digits():
    digits = datasets.load_digits()
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Training: {label}")
    return digits

def preprocess_data(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def train_classifier_with_tuning(data, target):
    dev_sizes = [0.3, 0.4, 0.5]
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1]
    }

    best_score = 0
    best_clf = None
    best_dev_size = None

    for dev_size in dev_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=dev_size, shuffle=True, random_state=42
        )

        grid = GridSearchCV(svm.SVC(), param_grid, cv=5)
        grid.fit(X_train, y_train)
        score = grid.score(X_test, y_test)

        if score > best_score:
            best_score = score
            best_clf = grid.best_estimator_
            best_dev_size = dev_size

    print("Best Dev Size:", best_dev_size)
    print("Best Parameters:", best_clf.get_params())
    print("Best Score:", best_score)

    # Return train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=best_dev_size, shuffle=True, random_state=42
    )
    return best_clf, X_train, X_test, y_train, y_test

def predict(clf, X_test):
    return clf.predict(X_test)

def visualize_predictions(X_test, predicted):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def print_classification_report(clf, y_test, predicted):
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, predicted)}\n")

def plot_confusion_matrix(y_test, predicted):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp

def rebuild_classification_report_from_cm(cm):
    y_true = []
    y_pred = []
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]
    print("Classification report rebuilt from confusion matrix:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")
