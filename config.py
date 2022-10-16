import random

import numpy as np
import torch,os
from torch.backends import cudnn

# Random seed to maintain reproducible results
import config
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
model_arch_name = "msrn"
# Model in channels
in_channels = 3
# Model in channels
out_channels = 3
# Image magnification factor
upscale_factor = 2
outputs_dir = os.path.join('network', 'x{}'.format(upscale_factor))
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = 'msrn'

if mode == "train":
    # Dataset address
    train_gt_images_dir = "./data/train"

    test_gt_images_dir = "./data/test/hr"
    test_lr_images_dir = f"./data/test/lr"

    gt_image_size = int(upscale_factor * 64)
    batch_size = 16
    num_workers = 1

    # Load the address of the pretrained model
    pretrained_model_weights_path =  os.path.join(config.outputs_dir, "model.pth")

    # Incremental training and migration training
    resume = ""

    # Total num epochs (1,000,000 iters)
    epochs = 100

    # loss function weights
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy (200,000 iters)
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 10  ###训练集多久保持/打印一次信息

if mode == "test":
    # Test data address
    test_gt_images_dir = "./data/test/hr"
    test_lr_images_dir = f"./data/test/lr"
    sr_dir = f"./results/sr/set5{upscale_factor}"

    model_weights_path = "./results/best.pth"
