from orpheus import OrpheusTrainer
model_name = "amuvarma/3b-10m-pretrain-full" # from stage_2_train.py

#** loading the datasets can take a while, even up to an hour **
orpheus = OrpheusTrainer(    
    stage = "stage_3",
    model_name = model_name,
    batch_size = 32, # use batch_size * number_of_gpus = 64 for quickest training
    )

orpheus_trainer = orpheus.create_trainer( report_to="wandb" )

orpheus_trainer.train() # subclasses 🤗 Trainer 