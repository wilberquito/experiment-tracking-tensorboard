import torchvision
from torch import nn
from going_modular.utils import set_seeds


def model_factory(model_name: str, output_shape: int, device: str = "cpu"):
    if model_name == "effnetb0":
        model = create_effnetb0(output_shape, device)
    elif model_name == "effnetb2":
        model = create_effnetb2(output_shape, device)
    else:
        raise Exception(f"[Error]: Unknown model - {model_name}")
    return model


# Create an EffNetB0 feature extractor
def create_effnetb0(output_shape: int, device: str = "cpu"):
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2), nn.Linear(in_features=1280, output_shape=output_shape)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model


# Create an EffNetB2 feature extractor
def create_effnetb2(output_shape: int, device: str = "cpu"):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features=1408, output_shape=output_shape)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model
