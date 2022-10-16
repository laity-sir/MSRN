import os
import queue
import threading

import cv2
import numpy as np
import torch
import config
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import imgproc


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]
class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        image_dir (str): 训练集或者验证集的地址
        gt_image_size (int): 高分辨率图像的大小
        upscale_factor (int):尺度因子
        mode (str): 数据导入的方式。训练集需要数据增强，测试集不需要数据增强
    """
    def __init__(self, image_dir: str, gt_image_size: int, upscale_factor: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        # Load training dataset or test dataset
        self.mode = mode

    def __getitem__(self, batch_index):
        #读取数据
        gt_image = cv2.imread(self.image_file_names[batch_index]).astype(np.float32) / 255.
        # if config.in_channels==1:
        #     gt_image=convert_rgb_to_y(gt_image)
        # Image processing operations
        if self.mode == "train":
            gt_image = imgproc.random_crop(gt_image, self.gt_image_size)
        elif self.mode == "val":
            gt_image = imgproc.center_crop(gt_image, self.gt_image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        gt_image = imgproc.random_rotate(gt_image, [90, 180, 270])
        gt_image = imgproc.random_horizontally_flip(gt_image, 0.5)
        gt_image = imgproc.random_vertically_flip(gt_image, 0.5)
        lr_image = imgproc.image_resize(gt_image, 1 / self.upscale_factor)
        # BGR convert to RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)
        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)

class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_gt_images_dir (str): 高分辨率图片
        test_lr_images_dir (str): 低分辨率图片
    """

    def __init__(self, test_gt_images_dir: str,test_lr_images_dir,upscale_factor) -> None:
        super(TestImageDataset, self).__init__()
        self.upscale_factor=upscale_factor
        # Get all image file names in folder
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in os.listdir(test_gt_images_dir)]
        self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in os.listdir(test_lr_images_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.
        # if config.in_channels==1:
        #     gt_image=convert_rgb_to_y(gt_image)
        #     lr_image=convert_rgb_to_y(lr_image)
        # lr_image = imgproc.image_resize(gt_image, 1 / self.upscale_factor)
        bic_image=imgproc.image_resize(lr_image, self.upscale_factor)

        # BGR convert to RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        bic_image = cv2.cvtColor(bic_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)
        bic_tensor = imgproc.image_to_tensor(bic_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor,'bicubic':bic_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)

def test_traindataset():
    """
    测试Trainvaldataset
    """
    dataset=TrainValidImageDataset(image_dir='./data/train',gt_image_size=48,upscale_factor=2,mode='train')
    print(len(dataset))
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from torchvision import transforms
    num=72
    col=8
    row=int(num/8)
    print(dataset[0]['lr'])
    index = np.random.randint(1, len(dataset), num)
    print(index)
    hh=[dataset[i]['gt'] for i in index]
    ###可视化图片
    for i in range(num):
        for j in range(8):
            # plt.figure()
            plt.subplot(row, col, i+1)
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(transforms.ToPILImage()(hh[i]), cmap='gray')
    plt.show()

def test_testdataset():
    class TestImageDataset(Dataset):
        """Define Test dataset loading methods.

        Args:
            test_gt_images_dir (str): 高分辨率图片
            test_lr_images_dir (str): 低分辨率图片
        """

        def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
            super(TestImageDataset, self).__init__()
            # Get all image file names in folder
            self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in os.listdir(test_gt_images_dir)]
            self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in os.listdir(test_lr_images_dir)]

        def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
            # Read a batch of image data
            gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32) / 255.
            lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.

            # BGR convert to RGB
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

            # Convert image data into Tensor stream format (PyTorch).
            # Note: The range of input and output is between [0, 1]
            gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
            lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

            return {"gt": gt_tensor, "lr": lr_tensor}

        def __len__(self) -> int:
            return len(self.gt_image_file_names)

    dataset=TestImageDataset(test_gt_images_dir='./data/test/hr',test_lr_images_dir='./data/test/lr')
    print(len(dataset))
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms


    for i in range(len(dataset)):
        plt.imshow(transforms.ToPILImage()(dataset[i]['gt']), cmap='gray')
        plt.show()

def test1():
    dataset=TestImageDataset(test_gt_images_dir='./data/test/hr',test_lr_images_dir='./data/test/lr',upscale_factor=3)
    print(len(dataset))
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms


    for i in range(len(dataset)):
        plt.imshow(transforms.ToPILImage()(dataset[i]['bicubic']), cmap='gray')
        plt.show()

if __name__=='__main__':
    test_traindataset()


