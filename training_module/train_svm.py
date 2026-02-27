from sklearn import svm
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import numpy as np

from audio_dataset_transformation_config import get_mfcc_transformation, get_spectrogram_transformation, \
    get_mel_transformation
from dataset.svm_dataset import AudioDatasetForSVM
from training_module.dataloader import get_dataloader


def extract_features_for_svm(dataloader):
    X = []
    y = []
    print("Extracting features for SVM...")
    for features, label in tqdm(dataloader):
        # Since batch_size might be > 1, iterate through the batch
        X.append(features.numpy())
        y.append(label.numpy())

    # Flatten batches and convert to 2D array: (Total Samples, Features)
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y


def train_svm(train_loader, val_loader, feature_name:str):
    # 1. Prepare Data
    X_train, y_train = extract_features_for_svm(train_loader)
    X_val, y_val = extract_features_for_svm(val_loader)

    # 2. Define and Train
    svm_classifier = svm.SVC(kernel='rbf', C=1.0)
    svm_classifier.fit(X_train, y_train)

    # 3. Calculate Accuracies
    # .score() returns the mean accuracy on the given data and labels
    train_acc = svm_classifier.score(X_train, y_train)
    val_acc = svm_classifier.score(X_val, y_val)

    # 4. Calculate F1 (as you did for the RNN)
    y_pred = svm_classifier.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"--- SVM Results metrics --- {feature_name}")
    print(f"Training Accuracy: {train_acc * 100:.2f}% | Validation Accuracy: {val_acc * 100:.2f}% | F1-score: {f1:.2f}")
    return svm_classifier

if __name__ == "__main__":
    transformation_list=[get_mfcc_transformation(),get_mel_transformation(),get_spectrogram_transformation()]
    transformation_name=["MFCC", "Mel Spectrogram", "Spectrogram"]
    for transformation, transformation_name  in zip (transformation_list, transformation_name):
        print(f"Training SVM using {transformation_name}...")
        train_loader, validation_loader =get_dataloader(transformation,dataset_class=AudioDatasetForSVM)
        train_svm(train_loader, validation_loader,feature_name=transformation_name)