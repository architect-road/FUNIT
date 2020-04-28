import torch
import os
import sys
import argparse
import shutil
from utils import get_config,get_train_loaders
from trainer import Trainer


# Load experiment setting
config = get_config("configs/human_nezha.yaml")
trainer = Trainer(config)
trainer.cuda()
print(trainer.model.dis_a)

# loaders = get_train_loaders(config) # get datasets
# train_content_loader = loaders[0]
# train_class_loader = loaders[1]
# test_content_loader = loaders[2]
# test_class_loader = loaders[3]

# for it, (co_data, cl_data) in enumerate(zip(train_content_loader, train_class_loader)):
#     d_acc,d_loss = trainer.dis_update(co_data, cl_data, config)
#     g_acc,g_loss = trainer.gen_update(co_data, cl_data, config,multigpus=False)
#     torch.cuda.synchronize()
#     print('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))
#     print('D loss: %.2f\t G loss: %.2f' % (d_loss, g_loss))
#     input()
