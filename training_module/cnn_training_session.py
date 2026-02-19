from audio_dataset_transformation_config import get_mel_transformation, get_mfcc_transformation
from config import N_MELS, N_MFCC
from dataset.audio_dataset import AudioDataset
from model_architecture.model_architecture import AudioCNN
from model_architecture.rnn_model_architecture_spectrum_mfcc import RNN, LSTM, BiLSTM, GRU
from training_module.dataloader import get_dataloader
from training_module.train import draw_f1_score, train, collate_fn

def cnn_training_setup(transformation,curr_model_name):


        train_loader, val_loader = get_dataloader(
            transformation,
            collate_fn=collate_fn,
            dataset_class=AudioDataset
        )

        train(
            train_loader,
            val_loader,
            model=AudioCNN(),
            model_name_str=curr_model_name
        )

        draw_f1_score(curr_model_name)

if __name__ == "__main__":
    #rnn_spectrogram_training_setup(get_mel_transformation(), N_MELS)
    cnn_training_setup(get_mfcc_transformation(),"CNN_MFCC")
    cnn_training_setup(get_mel_transformation(),"CNN_Spectrogram")