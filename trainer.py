# Description: This file contains the code for training the model using the dataset and the model defined in the model.py file. The training is done using the PyTorch Lightning library.
# The training process involves loading the dataset, defining the model, configuring the training parameters, and training the model using the dataset.
# The training process is done using the PyTorch Lightning Trainer class, which handles the training loop, validation loop, and other training-related tasks.

# import necessary libraries and modules
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from tourism_dataset import tourismDataset
from toursit_tagger import touristTagger
from data_loading import dataLoader
from data_loading import dataLoader
from bert_model_loader import loadBertModel
def main():
    #Defining the max-number of token that can be input to the Bert.
    #We set the Max token size to 512 if any input exceeds this token count, it will be either truncated or will be split into smaller parts
    MAX_TOKEN_COUNT = 512
    train_df, val_df, num_classes = dataLoader()
    # Load the BERT tokenizer
    tokenizer, _ = loadBertModel()
    # Create instances of the dataset
    train_dataset = tourismDataset(train_df, tokenizer)
    test_dataset = tourismDataset(val_df, tokenizer)

    #Defining a callback that save the best model during training based on the validation loss.
    checkpoint_callback = ModelCheckpoint(
    dirpath="tourist_sentiment_analysis/checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    )

    # Example usage of the model and DataLoader
    N_EPOCHS = 5
    BATCH_SIZE = 16
    # Create DataLoaders with num_workers=0
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    #Calculating the necessary parameters for the model training
    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    # Initialize the model and DataLoader
    model = touristTagger(n_classes=num_classes, steps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS)

    #trainer = pl.Trainer(max_epochs=5)
    trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    callbacks=[checkpoint_callback],
    devices='auto',
    accelerator = 'auto',
    enable_progress_bar=True
    )

    trainer.fit(model, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
