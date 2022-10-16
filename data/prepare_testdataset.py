import argparse
import os
import numpy as np
import PIL.Image as pil_image
from PIL import Image

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.

def test(args):
    if args.mode=='y':
        print('生成y通道数据')
    for i, image_path in enumerate(sorted(os.listdir(args.images_dir))):
        basename = image_path.split('/')[-1]
        basename = basename.split('.')[0]  ###baby
        hr = pil_image.open(os.path.join(args.images_dir,image_path)).convert('RGB')
        if args.mode=='y':  ###这里是生成y通道数据
            hr=np.array(hr)
            hr = convert_rgb_to_y(hr).astype(np.uint8)
            hr=Image.fromarray(hr)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr.save(os.path.join('./test/hr/', '{}.png'.format(basename)))
        lr.save(os.path.join('./test/lr/','{}.png'.format(basename)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='./Set5')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--mode', type=str,help='y:表示ycbcr',default='')
    args = parser.parse_args()
    test(args)
