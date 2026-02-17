from audio_dataset_transformation_config import get_mel_transformation
from config import N_MELS
from dataset.rnn_dataset import AudioDatasetSpectogram
from model_architecture.rnn_model_architecture_for_spectrum import RNN_Spectogram, LSTM_Spectogram, BiLSTM_Spectogram, \
    GRU_Spectogram
from training_module.dataloader import get_dataloader
from training_module.train import draw_f1_score, train, collate_fn


def rnn_spectrogram_training_setup():
    model_classes = [
        RNN_Spectogram,
        LSTM_Spectogram,
        BiLSTM_Spectogram,
        GRU_Spectogram
    ]

    for model_class in model_classes:
        curr_model_name = model_class.__name__
        print(f"\n--- Starting Training: {curr_model_name} ---")

        train_loader, val_loader = get_dataloader(
            get_mel_transformation(),
            collate_fn=collate_fn,
            dataset_class=AudioDatasetSpectogram
        )

        train(
            train_loader,
            val_loader,
            model=model_class(input_size=N_MELS),
            model_name_str=curr_model_name
        )

        draw_f1_score(curr_model_name)
if __name__ == "__main__":
    rnn_spectrogram_training_setup()