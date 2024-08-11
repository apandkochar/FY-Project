import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import random_split
from torchvision import transforms

from dataloader import CustomImageDataset
from model import Blurnet

print(pl.__version__)
IM_SIZE = 96

train_transforms = transforms.Compose(
    [
        transforms.Resize(IM_SIZE * 5),
        transforms.RandomCrop(IM_SIZE),
        transforms.ToTensor(),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.Resize(IM_SIZE * 5),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),
    ]
)

train_dataset = CustomImageDataset(
    annotations_file="train_set.csv", transform=train_transforms
)
test_dataset = CustomImageDataset(
    annotations_file="test_set.csv", transform=test_transforms
)

test_set, validation_dataset = random_split(
    test_dataset,
    [int(len(test_dataset) * 0.9), len(test_dataset) - int(len(test_dataset) * 0.9)],
)
validation_dataset.transforms = test_transforms

model = Blurnet(train_dataset, validation_dataset, test_dataset)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="sample-blurnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)



early_callback = EarlyStopping(
    monitor="val_loss", check_on_train_epoch_end=True, patience=10
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_callback],
    min_epochs=30,
    log_every_n_steps=2,
)

trainer.fit(model)
trainer.test(model)
