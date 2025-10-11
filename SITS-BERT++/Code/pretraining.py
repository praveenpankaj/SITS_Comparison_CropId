import torch
from dataset.dataset_wrapper import DataSetWrapper
from model import SBERT
from trainer import SBERTTrainer
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Set random seed
setup_seed(123)

class Config(object):
    dataset_path = 'GrapeTrainingData.csv'
    pretrain_path = 'checkpoints/pretrain/'
    valid_rate = 0.03
    max_length = 64
    num_features = 10
    epochs = 80
    batch_size = 512
    hidden_size = 256
    layers = 3
    attn_heads = 8
    learning_rate = 1e-4
    warmup_epochs = 10
    decay_gamma = 0.99
    dropout = 0.1
    gradient_clipping = 5.0

if __name__ == "__main__":
    config = Config()

    print("Loading training and validation data sets...")
    dataset = DataSetWrapper(batch_size=config.batch_size,
                             valid_size=config.valid_rate,
                             data_path=config.dataset_path,
                             num_features=config.num_features,
                             max_length=config.max_length,
                             apply_time_warping=True)

    train_loader, valid_loader = dataset.get_data_loaders()

    print("Initialing SITS-BERT...")
    sbert = SBERT(config.num_features, hidden=config.hidden_size, n_layers=config.layers,
                  attn_heads=config.attn_heads, dropout=config.dropout)

    trainer = SBERTTrainer(sbert, config.num_features,
                           train_dataloader=train_loader,
                           valid_dataloader=valid_loader,
                           lr=config.learning_rate,
                           warmup_epochs=config.warmup_epochs,
                           decay_gamma=config.decay_gamma,
                           gradient_clipping_value=config.gradient_clipping)

    print("Pre-training SITS-BERT...")
    mini_loss = np.Inf
    for epoch in range(config.epochs):
        train_loss, valida_loss = trainer.train(epoch)
        if mini_loss > valida_loss:
            mini_loss = valida_loss
            trainer.save(epoch, config.pretrain_path)