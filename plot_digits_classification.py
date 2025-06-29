from utils import (
    load_and_visualize_digits,
    preprocess_data,
    train_classifier_with_tuning,
    predict,
    visualize_predictions,
    print_classification_report,
    plot_confusion_matrix,
    rebuild_classification_report_from_cm
)
import matplotlib.pyplot as plt

def main():
    digits = load_and_visualize_digits()
    data, target = preprocess_data(digits)
    clf, X_train, X_test, y_train, y_test = train_classifier_with_tuning(data, target)
    predicted = predict(clf, X_test)
    visualize_predictions(X_test, predicted)
    print_classification_report(clf, y_test, predicted)
    disp = plot_confusion_matrix(y_test, predicted)
    rebuild_classification_report_from_cm(disp.confusion_matrix)
    plt.show()

if __name__ == "__main__":
    main()
