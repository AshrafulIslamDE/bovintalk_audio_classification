from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

from audio_dataset_transformation_config import get_mfcc_transformation, get_spectrogram_transformation, \
    get_mel_transformation
from dataset.svm_dataset import AudioDatasetForTraditionalMLAlgo
from training_module.dataloader import get_dataloader


def get_model():
    # 1. SVM
    svm_clf = svm.SVC(kernel='rbf', C=1.0)

    # 2. Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 3. k-Nearest Neighbors (k=2  for our case)
    knn_clf = KNeighborsClassifier(n_neighbors=2)

    # 4. Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000)

    return [svm_clf, rf_clf, knn_clf, lr_clf]

def extract_features_for_ml(dataloader):
    X = []
    y = []
    for features, label in tqdm(dataloader):
        # Since batch_size might be > 1, iterate through the batch
        X.append(features.numpy())
        y.append(label.numpy())

    # Flatten batches and convert to 2D array: (Total Samples, Features)
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

def train_model(train_loader, val_loader,classifier, feature_name:str):
    # 1. Prepare Data
    X_train, y_train = extract_features_for_ml(train_loader)
    X_val, y_val = extract_features_for_ml(val_loader)

    # Apply Scaling (Crucial for k-NN and SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 2. Define and Train
    classifier.fit(X_train, y_train)

    # 3. Calculate Accuracies
    # .score() returns the mean accuracy on the given data and labels
    train_acc = classifier.score(X_train, y_train)
    val_acc = classifier.score(X_val, y_val)

    # 4. Calculate F1 (as you did for the RNN)
    y_pred = classifier.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"--- {classifier.__class__.__name__} Results metrics --- {feature_name}")
    print(f"Training Accuracy: {train_acc * 100:.2f}% | Validation Accuracy: {val_acc * 100:.2f}% | F1-score: {f1:.2f}")


if __name__ == "__main__":
    transformation_list = [get_mfcc_transformation(), get_mel_transformation(), get_spectrogram_transformation()]
    transformation_names = ["MFCC", "Mel Spectrogram", "Spectrogram"]  # pluralized for clarity
    classifier_list = get_model()

    for classifier in classifier_list:
        for trans, t_name in zip(transformation_list, transformation_names):
            print(f"\n--- Starting Experiment: {classifier.__class__.__name__} + {t_name} ---")

            train_loader, validation_loader = get_dataloader(trans, dataset_class=AudioDatasetForTraditionalMLAlgo)

            train_model(
                train_loader,
                validation_loader,
                classifier=classifier,
                feature_name=t_name
            )