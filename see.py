import torch

scheduler_state = torch.load('./mymodel/checkpoint-200/scheduler.pt')
print(scheduler_state)