from audio_dataset_transformation_config import get_mel_transformation, get_mfcc_transformation
from dataset.audio_dataset import AudioDataset
from model_architecture.model_architecture import AudioCNN
from training_module.dataloader import get_dataloader
from training_module.train import draw_f1_score, train, collate_fn
from training_module.training_config_utils import update_config_from_args


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


if __name__ == "__main__":
    update_config_from_args()
    cnn_training_setup(get_mfcc_transformation(),"CNN_MFCC")
    cnn_training_setup(get_mel_transformation(),"CNN_Mel_Spectrogram")