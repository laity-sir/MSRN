import argparse
import os
import cv2
import numpy as np
import torch
from model import MSRN
import config
import imgproc
import model
import matplotlib.pyplot as plt
from torchvision import  transforms

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))

def show_tensor_img(tensor_img):
    to_pil = transforms.ToPILImage()
    img1 = tensor_img.cpu().clone()
    if img1.dim()==4:
        img1=img1.squeeze(0)
    img1 = to_pil(img1)
    hh=np.array(img1)
    plt.imshow(img1,cmap='gray')
    plt.show()

def main(args):
    # Initialize the super-resolution msrn_model
    msrn_model = MSRN(in_channels=config.in_channels,out_channels=config.out_channels,upscale_factor=config.upscale_factor)
    msrn_model = msrn_model.to(device=config.device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load the super-resolution msrn_model weights
    checkpoint = torch.load(args.model_weights_path, map_location=lambda storage, loc: storage)
    msrn_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    msrn_model.eval()

    # Read LR image and HR image
    lr_image = cv2.imread(args.inputs_path).astype(np.float32) / 255.0

    # Convert BGR channel image format data to RGB channel image format data
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_tensor = lr_tensor.to(device=config.device, non_blocking=True)
    # hh=imgproc.tensor_to_image(lr_tensor,True,False)
    # show_tensor_img(lr_tensor)


    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = msrn_model(lr_tensor)
        # show_tensor_img(sr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(args.output_path, sr_image)
    cv2.imshow('img',sr_image)
    cv2.waitKey(0)

    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the MSRN model generator super-resolution images.")
    parser.add_argument("--model_arch_name", type=str, default="msrn")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--inputs_path", type=str, default="./figure/comic_lr.png", help="Low-resolution image path.")
    parser.add_argument("--output_path", type=str, default="./figure/comic_sr.png", help="Super-resolution image path.")
    parser.add_argument("--model_weights_path", type=str,
                        default="./results/best.pth",
                        help="Model weights file path.")
    args = parser.parse_args()

    main(args)
