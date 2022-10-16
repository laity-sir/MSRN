import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import config
import model
from model import MSRN
from dataset import CUDAPrefetcher,CPUPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))

def main():
    resume=False
    # Initialize the number of training epochs
    start_epoch = 0
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0
    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    msrn_model, ema_msrn_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(msrn_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")
    save_data={
    }

    print("Check whether the pretrained model is restored...")
    if resume:
        msrn_model, ema_msrn_model, start_epoch, optimizer, scheduler = load_state_dict(
            msrn_model,
            config.pretrained_model_weights_path,
            ema_msrn_model,
            start_epoch,
            optimizer,
            scheduler,
            "resume")
        save_data = torch.load('./figure/save_data.pth')
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results

    results_dir = 'results'
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("results", "logs{}".format(config.upscale_factor)),flush_secs=60)

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device)
    ssim_model = ssim_model.to(device=config.device)

    for epoch in range(start_epoch, config.epochs):
        train_loss=train(msrn_model,ema_msrn_model,train_prefetcher,criterion,optimizer,epoch,scaler,writer)
        psnr, ssim,test_loss = validate(msrn_model,test_prefetcher,criterion,epoch,writer,psnr_model,ssim_model,"Test")
        save_data[epoch] = {'train_loss': train_loss, 'test_loss': test_loss, 'test_psnr': psnr,
                            'test_ssim': ssim}
        torch.save(save_data, './figure/save_data.pth')
        # Update LR
        scheduler.step()
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        state={
            "epoch": epoch,
            "state_dict": msrn_model.state_dict(),
            "ema_state_dict": ema_msrn_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()}
        save_checkpoint(state_dict=state,
                        path=config.outputs_dir,
                        results_dir=results_dir,
                        is_best=is_best,
                        is_last=is_last)

def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_gt_images_dir,
                                            config.gt_image_size,
                                            config.upscale_factor,
                                            "train")
    length=100
    #train_datasets,_=torch.utils.data.random_split(train_datasets,[length,len(train_datasets)-length])
    test_datasets = TestImageDataset(config.test_gt_images_dir, config.test_lr_images_dir,config.upscale_factor)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    if torch.cuda.is_available():
        train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
        test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        test_prefetcher = CPUPrefetcher(test_dataloader)

    return train_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    # msrn_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,out_channels=config.out_channels)
    msrn_model=MSRN(in_channels=config.in_channels,out_channels=config.out_channels,upscale_factor=config.upscale_factor)
    msrn_model = msrn_model.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_msrn_model = AveragedModel(msrn_model, avg_fn=ema_avg)

    return msrn_model, ema_msrn_model


def define_loss() -> nn.L1Loss:
    criterion = nn.L1Loss()
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(msrn_model) -> optim.Adam:
    optimizer = optim.Adam(msrn_model.parameters(), config.model_lr, config.model_betas, config.model_eps)

    return optimizer


def define_scheduler(optimizer: optim.Adam) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)

    return scheduler


def train(
        msrn_model: nn.Module,
        ema_msrn_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("train_pro_time", ":6.3f")  ##处理一批数据的时间
    data_time = AverageMeter("train_in_data", ":6.3f") ###加载一批数据的时间
    losses = AverageMeter("train_loss", ":6.6f")  ###一批数据的loss
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    msrn_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)
        ###释放不用的变量
        del batch_data
        torch.cuda.empty_cache()

        # Initialize generator gradients
        msrn_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = msrn_model(lr)
            loss = torch.mul(config.loss_weights, criterion(sr, gt))

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        ###释放不用的变量
        del gt, lr,sr
        torch.cuda.empty_cache()
        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()
        # Update EMA
        ema_msrn_model.update_parameters(msrn_model)
        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)
        ###释放不用的变量
        del loss
        torch.cuda.empty_cache()
        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

    progress.display_summary()
    return losses.avg

def validate(
        msrn_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("val_time", ":6.3f")  ###每批数据处理的时间
    psnres = AverageMeter("val_psnr", ":4.2f")
    ssimes = AverageMeter("val_ssim", ":4.4f")
    losses = AverageMeter("val_loss", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes,losses], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    msrn_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            ###释放不用的变量
            del batch_data
            torch.cuda.empty_cache()

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = msrn_model(lr)
                loss = torch.mul(config.loss_weights, criterion(sr, gt))

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))
            losses.update(loss.item(),lr.size(0))
            ###释放不用的变量
            del gt, lr, loss,sr,psnr,ssim
            torch.cuda.empty_cache()

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()
            # Preload the next batch of data
            batch_data = data_prefetcher.next()
            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
        writer.add_scalar(f"{mode}/loss", losses.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg,losses.avg


if __name__ == "__main__":
    main()
