import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms


NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 2*(10**-5)


train_dir = "/content/car_data/train"
test_dir = "/content/car_data/test"


device = "cuda" if torch.cuda.is_available() else "cpu"

model,data_transform = model_builder.get_model(num_classes = 196
)
model = model.to(device)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)




loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)


utils.save_model(model=model,
                 target_dir="models",
                 model_name="best_model.pth")
