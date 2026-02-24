from audio_dataset_transformation_config import get_mel_transformation, get_mfcc_transformation
from config import N_MELS, N_MFCC
from dataset.rnn_dataset import AudioDatasetOfSpectogramAndMFCCForRNN
from model_architecture.rnn_model_architecture_spectrum_mfcc import RNN, LSTM, BiLSTM, GRU
from training_module.dataloader import get_dataloader
from training_module.train import draw_f1_score, train, collate_fn


def rnn_spectrogram_training_setup(transformation, input_size,input_feature_name):
    model_classes = [
        RNN,
        LSTM,
        BiLSTM,
        GRU
    ]

    for model_class in model_classes:
        curr_model_name = model_class.__name__+"_"+input_feature_name
        print(f"\n--- Starting Training: {curr_model_name} ---")

        train_loader, val_loader = get_dataloader(
            transformation,
            collate_fn=collate_fn,
            dataset_class=AudioDatasetOfSpectogramAndMFCCForRNN
        )

        train(
            train_loader,
            val_loader,
            model=model_class(input_size=input_size),
            model_name_str=curr_model_name
        )

        draw_f1_score(curr_model_name)
if __name__ == "__main__":
    rnn_spectrogram_training_setup(get_mel_transformation(), N_MELS,"Spectrogram")
    rnn_spectrogram_training_setup(get_mfcc_transformation(), N_MFCC,"MFCC")