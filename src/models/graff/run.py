import time
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import logging
from sacred import Experiment
import seml
from seml.utils import flatten

from src.datasets import get_dataset
from src.models.graff.GRAFF import GRAFF

ex = Experiment()
seml.setup_logger(ex)

project_name = 'dynamic-graff'

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(# Dataset parameters,
        seed_dataset=123,
        directory_dataset="/nfs/staff-hdd/charpent/datasets/",
        dataset_name="cora",
        split_type="geom-gcn",
        split_index=0,
        split_ratio=[.9, .1, .1],

        # Model parameters
        directory_model="/nfs/staff-hdd/charpent/dynamic-graff/",
        hidden_dim=512,
        n_encoder_layers=1,
        n_layers=2,
        n_decoder_layers=2,
        W_type="diag_dom",
        omega_type="scalar",
        Q_type="zero",
        tau=1.,
        dropout=0.,
        seed_model=123,

        # Training parameters
        lr=1e-3,
        weight_decay=0,
        patience=50,
        max_epochs=10000,
        ):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###################
    ## Load datasets ##
    ###################
    data_module = get_dataset[dataset_name](data_dir=directory_dataset,
                                            split_type=split_type,
                                            split_index=split_index,
                                            split_ratio=split_ratio,
                                            seed=seed_dataset,)
    data_module.setup()
    input_dim, output_dim = data_module.input_dim, data_module.output_dim

    #################
    ## Train model ##
    #################
    params_dict= {'input_dim': input_dim,
                  'hidden_dim': hidden_dim,
                  'output_dim': output_dim,
                  'n_encoder_layers': n_encoder_layers,
                  'n_layers': n_layers,
                  'n_decoder_layers': n_decoder_layers,
                  'W_type': W_type,
                  'omega_type': omega_type,
                  'Q_type': Q_type,
                  'tau': tau,
                  'dropout': dropout,
                  "lr": lr,
                  "weight_decay": weight_decay,
                  'seed': seed_model}
    model = GRAFF(**params_dict)

    random_name = str(random.randint(0, 1e6))
    model_path = f"{directory_model}/gcn-{random_name}"
    while os.path.exists(model_path):
        random_name = str(random.randint(0, 1e6))
        model_path = f"{directory_model}/gcn-{random_name}"
    os.makedirs(model_path)

    early_stopping = EarlyStopping(monitor='val_acc',
                                   patience=patience,
                                   mode="max",
                                   check_finite=True)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                          save_top_k=1,
                                          mode="max",
                                          every_n_epochs=1,
                                          dirpath=model_path,
                                          filename='model-{epoch:02d}-{validation_loss:.2f}')
    wandb_logger = WandbLogger(save_dir=model_path,
                               name=f'{project_name}-logger-{random_name}',
                               project=project_name)
    t0 = time.time()
    trainer = pl.Trainer(callbacks=[early_stopping, checkpoint_callback],
                         gpus=1,
                         max_epochs=max_epochs,
                         logger=wandb_logger)
    trainer.fit(model, data_module)
    t1 = time.time()

    ################
    ## Test model ##
    ################
    best_model_path = checkpoint_callback.best_model_path
    # model = model.load_from_checkpoint(best_model_path)
    # trainer.test(test_dataloaders=data_module.test_dataloader(), ckpt_path=best_model_path) # TODO: does not work for no reason.
    results = trainer.test(dataloaders=data_module.test_dataloader(), ckpt_path=best_model_path)[0]

    fail_trace = {
        'fail_trace': seml.evaluation.get_results,
        'best_model_path': best_model_path,
        'training_time': t1 - t0,
    }

    return {**results, **fail_trace}
