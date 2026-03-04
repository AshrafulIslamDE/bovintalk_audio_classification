from torchvision import models
from torch import nn

from audio_dataset_transformation_config import get_mfcc_transformation, get_mel_transformation
from dataset.transfer_learning_dataset import TransferLearningDataset
from training_module.dataloader import get_dataloader
from training_module.train import draw_f1_score, train, collate_fn
from training_module.training_config_utils import update_config_from_args


def download_pretrained_model()->list[nn.Module]:
    # ------------------------
    # Load pretrained Model
    # ------------------------
    vgg16 = models.vgg16(weights="IMAGENET1K_V1")
    resnet18 = models.resnet18(weights="IMAGENET1K_V1")

    # Replace classifier for 2 classes
    vgg16.classifier[6] = nn.Linear(4096, 2)

    resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)

    return [resnet18,vgg16]

def finetune_downloaded_model(model:nn.Module)->nn.Module:
    # Freeze feature extractor
    if model is models.vgg16:
        for p in model.features.parameters():
            p.requires_grad = False
    elif model is models.resnet18:
        for p in model.parameters():
            p.requires_grad = False  # freeze everything

        # Unfreeze final fc layer
        for p in model.fc.parameters():
            p.requires_grad = True


    return model

def prepare_model()-> list[nn.Module]:
    model_list=download_pretrained_model()
    for idx,model in enumerate(model_list):
        model=finetune_downloaded_model(model)
        model_list[idx]=model
    return model_list

def start_training(model:nn.Module,transformation:nn.Module, model_name:str)->None:
    print(f"\n--- Starting Training: {model_name} ---")

    train_loader, val_loader = get_dataloader(
        transformation,
        collate_fn=collate_fn,
        dataset_class=TransferLearningDataset
    )

    train(
        train_loader,
        val_loader,
        model=model,
        model_name_str=model_name
    )

if __name__=="__main__":
    update_config_from_args()
    model_list=prepare_model()
    for model in model_list:
        start_training(model, get_mfcc_transformation(), model.__class__.__name__ + "_mfcc")
        start_training(model, get_mel_transformation(), model.__class__.__name__ + "_spectrogram")





