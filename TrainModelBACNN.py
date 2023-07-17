def main(training_parameters_name, num_gpus, num_nodes):

    """ Setting CUDA restrictions"""

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50" # Control caching allocater
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 # Disable caching


    """ Login to Weight and Biases"""
    wandb.login(key='')

    """ Create required folders """

    DATA_DIR = Path(__file__).parent.parent / 'data'
    TRAINING_PARAMETERS = Path(__file__).parent / f'json_parameters/{training_parameters_name}.json'
    
    SAVED_MEAN_STD = Path(__file__).parent / 'libraries/saved_mean_std'
    EXPERIMENT_FOLDER = Path(__file__).parent / f'experiments_folder/{training_parameters_name}'
    
    MODELS_DIR = EXPERIMENT_FOLDER / f"models"
    BEST_MODEL_DIR = MODELS_DIR / "best_model"
    MODEL_CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
    LIGHTNING_LOGS = MODELS_DIR / 'lightning_logs'
    SAVE_MODEL_DIR = MODELS_DIR / "saved_models"
    
    RESULTS = EXPERIMENT_FOLDER / "results"
    last_checkpoint = MODEL_CHECKPOINT_DIR / 'last.ckpt'

    MODELS_DIR.mkdir(exist_ok=True)
    BEST_MODEL_DIR.mkdir(exist_ok=True)
    MODEL_CHECKPOINT_DIR.mkdir(exist_ok=True)
    LIGHTNING_LOGS.mkdir(exist_ok=True)
    SAVE_MODEL_DIR.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)


    """ Load parameters """

    f = open(TRAINING_PARAMETERS, "rb")
    parameters = json.load(f)

    """ model parameters """

    model_name = parameters["model_name"]
    num_classes = np.array(parameters["num_classes"])
    results = RESULTS
    #pretrained = parameters["pretrained"]
    #fine_tune = parameters["fine_tune"]
    #dropout = parameters["dropout"]
    learning_rate = parameters["learning_rate"]
    learning_rate_finder = parameters["learning_rate_finder"]
    experiment_folder = parameters["experiment_folder"]
    
    attention_size = parameters["attention_size"]
    feature_extractor_path = parameters["feature_extractor_path"]
    freeze_feature_extractor = parameters["freeze_feature_extractor"]
    freeze_feature_block1 = parameters["freeze_feature_block1"]
    freeze_feature_block2 = parameters["freeze_feature_block2"]
    
    loss_weights0 = parameters["loss_weights0"]
    loss_weights1 = parameters["loss_weights1"]
    loss_weights2 = parameters["loss_weights2"]
    loss_weights3 = parameters["loss_weights3"]
    
    num_epochs0 = parameters["num_epochs0"]
    num_epochs1 = parameters["num_epochs1"]
    num_epochs2 = parameters["num_epochs2"]
    num_epochs3 = parameters["num_epochs3"]

    
    scheduler = parameters["scheduler"]
    scheduler_parameters = parameters["scheduler_parameters"]
    #loss_function = parameters["loss_function"]

    """ datamodule parameters """

    dataset_name = parameters["dataset_name"]
    image_folder = parameters["image_folder"]
    mean_std_name = parameters["mean_std_name"]
    img_size = parameters["img_size"]
    seed_value = parameters["seed_value"]
    batch_size = parameters["batch_size"]
    sampler = parameters["sampler"]
    
    manual_augmentations = parameters["manual_augmentations"]
    number_of_augmentations = parameters["number_of_augmentations"]
    magnitude_of_augmentations = parameters["magnitude_of_augmentations"]



    seed_everything(seed_value)

    DATASET_DIR = DATA_DIR / image_folder
    MEAN_STD_PATH = SAVED_MEAN_STD / mean_std_name

    """ Create and initialise dataloaders """
    
    print(f'\nNumber of devices is: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i} name is: {torch.cuda.get_device_name(f"cuda:{i}")}')
    
    


    dataModule = ImageFolderLightningDataModule(
                                                        data_dir=DATASET_DIR,
                                                        mean_std_path = MEAN_STD_PATH,
                                                        image_size = img_size,
                                                        batch_size = batch_size,
                                                        split_seed = seed_value,
                                                        num_workers=cpu_count(), 
                                                        sampler = sampler,                                             # Use True for Weighted Sampler or False to not use any
                                                        pin_memory = True,   # Default is False
                                                        manual_augmentations = manual_augmentations,
                                                        number_of_augmentations = number_of_augmentations,
                                                        magnitude_of_augmentations = magnitude_of_augmentations,
                                                    )

    dataModule.setup()

    """ Build Model """

    model = BA_CNN_Lightning(
                            results = RESULTS,
                            model_name = model_name,
                            num_classes = num_classes,
                            attention_size = attention_size,
                            feature_extractor_path=feature_extractor_path,
                            freeze_feature_extractor=freeze_feature_extractor,
                            freeze_feature_block1=freeze_feature_block1,
                            freeze_feature_block2=freeze_feature_block2,
                            learning_rate = learning_rate,
                            scheduler=scheduler,
                            scheduler_parameters=scheduler_parameters,
                            loss_weights=loss_weights0,
                        )


    """ Configure Trainer """
    
    devices = num_gpus
    accelerator = 'cuda' #'gpu' 
    strategy = 'ddp' # "ddp" # "ddp_find_unused_parameters_false" #'ddp' #fsdp_native  'ddp' #sharded implementation using FairScale has been deprecated in v1.9.0 and will be removed in v2.0.0
    max_epochs = num_epochs3
    
    print(f'\nUsing {devices} devices')
    print(f'Using {num_nodes} nodes')
    print(f'Strategy is: {strategy}')
    print(f'Accelerator is: {accelerator}')
    print(f'\nMax epochs parameter is: {max_epochs}')

    print(f'\nUsing an initial learning rate of {learning_rate}\n')


    """ Configure Weights and Biaises """
    
    # Define a config dictionary object
    config = {
        "training_parameters_name": training_parameters_name,
        "dataset": dataset_name,
        "model": model_name,
        "manual_augmentations": manual_augmentations,
        "img_size": img_size,
        #"loss_function": loss_function,
        "image_folder": image_folder,
        "scheduler": scheduler,
        "learning_rate_finder": learning_rate_finder,
        "sampler": sampler,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        
        "loss_weights0": loss_weights0,
        "loss_weights1": loss_weights1,
        "loss_weights2": loss_weights2,
        "loss_weights3": loss_weights3,
        
        "num_epochs0": num_epochs0,
        "num_epochs1": num_epochs1,
        "num_epochs2": num_epochs2,
        "num_epochs3": num_epochs3,
        
        'freeze_feature_extractor': freeze_feature_extractor,
        'freeze_feature_block1': freeze_feature_block1,
        'freeze_feature_block2': freeze_feature_block2,
    }

    # Update the config dictionary when you initialize W&B
    run = wandb.init(
                    project='plankton-Paper3',
                    entity="oceania-plankton",
                    config=config,
                    )

    # If there is a last checkpoint then use it otherwise traing from zero
    if last_checkpoint.is_file() == True:
        ckpt_path = last_checkpoint
    else:
        ckpt_path = None
        
    """ Set callbacks """

    best_models_checkpoint_callback0 = ModelCheckpoint(
        dirpath=BEST_MODEL_DIR, save_top_k=1, verbose=False, monitor="val_1_f1", mode="max"
    )
    best_models_checkpoint_callback1 = ModelCheckpoint(
        dirpath=BEST_MODEL_DIR, save_top_k=1, verbose=False, monitor="val_2_f1", mode="max"
    )
    best_models_checkpoint_callback2 = ModelCheckpoint(
        dirpath=BEST_MODEL_DIR, save_top_k=1, verbose=False, monitor="val_3_f1", mode="max"
    )
    best_models_checkpoint_callback3 = ModelCheckpoint(
        dirpath=BEST_MODEL_DIR, save_top_k=1, verbose=False, monitor="val_3_f1", mode="max"
    )


    resume_checkpoint_callback = ModelCheckpoint(dirpath=MODEL_CHECKPOINT_DIR, save_last=True, save_on_train_epoch_end=True)

    early_stop_callback0 = EarlyStopping(monitor="val_1_f1", min_delta=0.0001, patience=30, verbose=True, mode="max")
    early_stop_callback1 = EarlyStopping(monitor="val_2_f1", min_delta=0.0001, patience=30, verbose=True, mode="max")
    early_stop_callback2 = EarlyStopping(monitor="val_3_f1", min_delta=0.0001, patience=30, verbose=True, mode="max")
    early_stop_callback3 = EarlyStopping(monitor="val_3_f1", min_delta=0.0001, patience=30, verbose=True, mode="max")

    swa_callback = StochasticWeightAveraging(swa_lrs=1e-3)

    checkpoint0 = ModelCheckpoint(monitor="val_1_f1", mode="min")
    checkpoint1 = ModelCheckpoint(monitor="val_2_f1", mode="min")
    checkpoint2 = ModelCheckpoint(monitor="val_3_f1", mode="min")
    checkpoint3 = ModelCheckpoint(monitor="val_3_f1", mode="min")

    wandb_logger = WandbLogger(
        #save_dir=str(MODELS_DIR),
        log_model=False)
    
    #Tensorboard_logger = TensorBoardLogger(save_dir='tb_logs', name=model_name)
    
    # Defining Training Parameters
    training_step_finished = [Path(RESULTS / f'Training_step_{i}_finished.txt').exists() for i in range(4)]
    num_epochs_list = [num_epochs0,
                       num_epochs1,
                       num_epochs2,
                       num_epochs3]
    
    loss_weights_list = [loss_weights0,
                         loss_weights1,
                         loss_weights2,
                         loss_weights3]
    
    best_models_checkpoint_callback_list = [best_models_checkpoint_callback0,
                                            best_models_checkpoint_callback1,
                                            best_models_checkpoint_callback2,
                                            best_models_checkpoint_callback3]
    
    early_stop_callback_list = [early_stop_callback0,
                                early_stop_callback1,
                                early_stop_callback2,
                                early_stop_callback3]
    
    checkpoint_list = [checkpoint0,
                       checkpoint1,
                       checkpoint2,
                       checkpoint3]
    is_lr_finder_done = False
    
    # Training Loop over every BT-step
    for training_step in range(4):
        model.update_loss_weights(loss_weights_list[training_step])
        if not training_step_finished[training_step]:
            trainer = Trainer(
                        devices=devices,
                        num_nodes=num_nodes,
                        accelerator=accelerator, 
                        max_epochs = num_epochs_list[training_step], 
                        strategy=strategy,
                        precision=32,
                        benchmark=True, 
                        auto_scale_batch_size = None,
                        log_every_n_steps=25, # Default is 50 training steps
                        default_root_dir= LIGHTNING_LOGS,
                        logger=[wandb_logger], #, Tensorboard_logger
                        accumulate_grad_batches= 3,
                        gradient_clip_val= 3,
                        gradient_clip_algorithm= 'value',
                        callbacks=[
                          checkpoint_list[training_step],
                          best_models_checkpoint_callback_list[training_step],
                          resume_checkpoint_callback,
                          early_stop_callback_list[training_step],
                          TQDMProgressBar(refresh_rate=29),
                          ]
                      )
            model.update_loss_weights(loss_weights_list[training_step])
            
            if learning_rate_finder and not is_lr_finder_done:
                print('\nINFO: Configuring Tuner for lr finder')
                tuner = Tuner(trainer=trainer)
                lr_finder = tuner.lr_find(model, datamodule=dataModule)
                optimized_learning_rate = lr_finder.suggestion()

                print(f'\nUsing an Optimal initial learning rate of: {optimized_learning_rate}\n')
                model.learning_rate = optimized_learning_rate
                is_lr_finder_done = True
            
            print(f'\nINFO: Starting Model training number {training_step}')
            print(f'\nINFO: Using weights {loss_weights_list[training_step]}')
            
            trainer.fit(model, datamodule=dataModule, ckpt_path=ckpt_path)
            
            print(f'\nINFO: Model training number {training_step} is finished')
            with open(RESULTS / f'Training_step_{training_step}_finished.txt', 'w') as f:
                f.write(f'Part "{training_step}" of training of experiment "{training_parameters_name}" is finished !')
                f.close()
        else:
            print(f'\nINFO: Training step number "{training_step}" is already finished. Starting Next training step')
    
    
    """ Save model """
    torch.save(model, SAVE_MODEL_DIR/ f'{model.model_name}_epoch={max_epochs}.pt')
    
    """ Finish logging to Weight and Biases """
    wandb.finish()
    
    

        
    """ Test Model """

    print('\nINFO: Starting Model testing')


    #torch.cuda.empty_cache()
    print(f'ckpt_path is {ckpt_path}')

    BEST_MODEL_CHECKPOINT = list(BEST_MODEL_DIR.glob('*.ckpt'))[0]
    print(f'best model ckpt is {BEST_MODEL_CHECKPOINT}')
    modelTest = BA_CNN_Lightning.load_from_checkpoint(ckpt_path)
    modelTest.eval()

    trainer_test = Trainer(devices=1,
                           num_nodes=1,
                           accelerator=accelerator,
                           callbacks=[TQDMProgressBar(refresh_rate=29)]
                          )
    dataModule.batch_size = 8

    test_results = trainer_test.test(modelTest, datamodule=dataModule)
    
    print('\nINFO: Model testing is finished')
    print(test_results)



if __name__ == "__main__":
    import sys
    sys.path.append('..')

    import argparse

    import json
    import wandb
    import logging
    import os
    from pathlib import Path
    os.chdir(Path(__file__).parent)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    
    
    from multiprocessing import cpu_count

    import torch
    import numpy as np
    import torchvision.transforms as tt
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import WandbLogger #, TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, StochasticWeightAveraging
    from pytorch_lightning.tuner.tuning import Tuner

    """ import my code: lightning classifier, lightning datamodule and TrainerArgs """ 

    from libraries.lightningDMBACNN import ImageFolderLightningDataModule
    from libraries.lightningModelBACNN import BA_CNN_Lightning


    """ import arguments """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--param-name")
    parser.add_argument("--num-gpus")
    parser.add_argument("--num-nodes")
    
    args = parser.parse_args()
    
    training_parameters_name = args.param_name
    num_gpus = int(args.num_gpus)
    num_nodes = int(args.num_nodes)
    
    main(training_parameters_name, num_gpus, num_nodes)
