from utils import save_model
from going_modular.utils import set_seeds
from typing import List
from torch.data.dataloader import DataLoader
from going_modular.models import model_factory
import torch
import torch.nn as nn
from going_modular.engine import train
from going_modular.utils import create_writer
from going_modular import data_setup
from going_modular.data_setup import download_data
from torchvision import transforms


OUT_FEATURES = 2
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download 10 percent and 20 percent training data (if necessary)
data_10_percent_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi",
)

data_20_percent_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
    destination="pizza_steak_sushi_20_percent",
)

# Setup training directory paths
train_dir_10_percent = data_10_percent_path / "train"
train_dir_20_percent = data_20_percent_path / "train"

# Setup testing directory paths (note: use the same test dataset for both to compare the results)
test_dir = data_10_percent_path / "test"

# Check the directories
print(f"Training directory 10%: {train_dir_10_percent}")
print(f"Training directory 20%: {train_dir_20_percent}")
print(f"Testing directory: {test_dir}")

# Setup ImageNet normalization levels (turns all images into similar distribution as ImageNet)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

simple_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 1. Resize the images
        transforms.ToTensor(),  # 2. Turn the images into tensors with values between 0 & 1
        normalize,  # 3. Normalize the images so their distributions match the ImageNet dataset
    ]
)

# Create 10% training and test DataLoaders
(
    train_dataloader_10_percent,
    test_dataloader,
    class_names,
) = data_setup.create_dataloaders(
    train_dir=train_dir_10_percent,
    test_dir=test_dir,
    transform=simple_transform,
    batch_size=BATCH_SIZE,
)

# Create 20% training and test data DataLoders
(
    train_dataloader_20_percent,
    test_dataloader,
    class_names,
) = data_setup.create_dataloaders(
    train_dir=train_dir_20_percent,
    test_dir=test_dir,
    transform=simple_transform,
    batch_size=BATCH_SIZE,
)

# Find the number of samples/batches per dataloader (using the same test_dataloader for both experiments)
print(
    f"Number of batches of size {BATCH_SIZE} in 10 percent training data: {len(train_dataloader_10_percent)}"
)
print(
    f"Number of batches of size {BATCH_SIZE} in 20 percent training data: {len(train_dataloader_20_percent)}"
)
print(
    f"Number of batches of size {BATCH_SIZE} in testing data: {len(train_dataloader_10_percent)} (all experiments will use the same test set)"
)
print(f"Number of classes: {len(class_names)}, class names: {class_names}")

# 1. Create an instance of models with pretrained weights
effnetb0 = model_factory("effnetb0", out_features=OUT_FEATURES, device=DEVICE)
effnetb2 = model_factory("effnetb2", out_features=OUT_FEATURES, device=DEVICE)

TRAIN_DATALOADERS = {
    "data_10_percent": train_dataloader_10_percent,
    "data_20_percent": train_dataloader_20_percent,
}


def roulette(
    train_dataloaders: List[DataLoader],
    test_dataloader: DataLoader,
    num_epochs: List[int],
    models: List[str],
    out_features: int,
    device: str = "cpu",
):
    # 1. Set the random seeds
    set_seeds(seed=42)

    # 2. Keep track of experiment numbers
    experiment_number = 0

    # 3. Loop through each DataLoader
    for dataloader_name, train_dataloader in train_dataloaders.items():
        # 4. Loop through each number of epochs
        for epochs in num_epochs:
            # 5. Loop through each model name and create a new model based on the name
            for model_name in models:
                # 6. Create information print outs
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] DataLoader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")

                # 7. Select the model
                model = model_factory(model_name, out_features, device)
                model.to(device)

                # 8. Create a new loss and optimizer for every model
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

                # 9. Train target model with target dataloaders and track experiments
                train(
                    model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=epochs,
                    device=device,
                    writer=create_writer(
                        experiment_name=dataloader_name,
                        model_name=model_name,
                        extra=f"{epochs}_epochs",
                    ),
                )

                # 10. Save the model to file so we can get back the best model
                save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
                save_model(model=model, target_dir="models", model_name=save_filepath)
                print("-" * 50 + "\n")
