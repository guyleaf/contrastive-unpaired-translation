import argparse
import os.path
import random
from operator import attrgetter

from PIL import Image

import util.util as util
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.

    You can also specify the relative path of domain folder with the domain flag '--data_domainA train/A --data_domainB train/B'.
    By default, domain folder will be set as the phase + 'A' or 'B'.
    """

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser, is_train: bool):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument(
            "--data_domainA",
            type=str,
            default=None,
            help="relative path of domain A folder to the dataroot",
        )
        parser.add_argument(
            "--data_domainB",
            type=str,
            default=None,
            help="relative path of domain B folder to the dataroot",
        )
        parser.add_argument(
            "--resampling",
            type=str,
            default=Image.Resampling.BICUBIC.name,
            choices=list(map(attrgetter("name"), Image.Resampling)),  # noqa: F821
            help="relative path of domain B folder to the dataroot",
        )
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = opt.phase + "A" if opt.data_domainA is None else opt.data_domainA
        self.dir_B = opt.phase + "B" if opt.data_domainB is None else opt.data_domainB

        self.dir_A = os.path.join(opt.dataroot, self.dir_A)
        self.dir_B = os.path.join(opt.dataroot, self.dir_B)

        if (
            opt.phase == "test"
            and not os.path.exists(self.dir_A)
            and os.path.exists(os.path.join(opt.dataroot, "valA"))
        ):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[
            index % self.A_size
        ]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(
            self.opt,
            load_size=self.opt.crop_size if is_finetuning else self.opt.load_size,
        )
        transform = get_transform(modified_opt, method=Image.Resampling[modified_opt.resampling])
        A = transform(A_img)
        B = transform(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
