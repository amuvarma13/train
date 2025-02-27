from orpheus import OrpheusTrainer
model_name = "checkpoints/checkpoint-7813" # from stage_2_train.py

#** loading the datasets can take a while, even up to an hour **
orpheus = OrpheusTrainer(    
    stage = "stage_4",
    model_name = model_name,
    batch_size = 21, # use batch_size * number_of_gpus = 64 for quickest training
    )

orpheus_trainer = orpheus.create_trainer( report_to="wandb" )

orpheus_trainer.train() # subclasses 🤗 Trainer