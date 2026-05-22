"""CUT-compatible adapter around `data.bilateral.UnpairedBilateralDataset`."""

import random

from data.base_dataset import BaseDataset
from data.bilateral import UnpairedBilateralDataset as _UnpairedBilateral


class UnpairedBilateralDataset(BaseDataset):
    """Wraps the unpaired L CC / R CC mammography dataset for CUT training.

    A → left CC, B → randomly sampled right CC. Length is the number of left
    images; right images are drawn uniformly per `__getitem__`, matching the
    unpaired assumption of CUT.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--annotations_csv', type=str, required=True,
                            help='Path to VinDr-Mammo finding_annotations.csv')
        parser.add_argument('--split', type=str, default='training',
                            choices=['training', 'test'],
                            help='VinDr split column to load')
        parser.add_argument('--flip_right', action='store_true',
                            help='Flip R-breast horizontally to match L orientation')
        parser.add_argument('--bilateral_size', type=int, nargs=2,
                            default=(512, 384), metavar=('H', 'W'),
                            help='Output image size as H W (must be multiples of 4)')
        parser.add_argument('--crop_width', type=int, default=360,
                            help='Crop output width to this many px after flip, '
                                 'keeping the chest-wall (right) edge. Must be a '
                                 'multiple of 4 and <= bilateral_size width. '
                                 '0 disables cropping.')
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='none')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.inner = _UnpairedBilateral(
            data_root=opt.dataroot,
            annotations_csv=opt.annotations_csv,
            split=opt.split,
            img_size=tuple(opt.bilateral_size),
            flip_right=opt.flip_right,
            crop_width=opt.crop_width,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        l_path = self.inner.left_images[index]
        r_path = random.choice(self.inner.right_images)
        a = self.inner._load(l_path, flip=False)
        b = self.inner._load(r_path, flip=self.inner.flip_right)
        return {'A': a, 'B': b, 'A_paths': str(l_path), 'B_paths': str(r_path)}
