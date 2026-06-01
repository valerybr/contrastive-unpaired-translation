"""CUT-compatible adapter around `data.bilateral.BilateralDataset`."""

from data.base_dataset import BaseDataset
from data.bilateral import BilateralDataset as _Bilateral, _load_mask, _mask_path


class BilateralDataset(BaseDataset):
    """Wraps the paired L CC / R CC mammography dataset for CUT training.

    A → left CC, B → right CC. The pairing info is unused by CUT (which is
    unpaired by construction) but kept so the same loader can be reused for
    paired evaluation.
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
        parser.add_argument('--finding_filter', type=str, default='no_finding',
                            choices=['no_finding', 'left_finding',
                                     'right_finding', 'either_finding'],
                            help='Which CC pairs to keep by finding label: '
                                 'no_finding (default, all-normal studies) or '
                                 'left/right/either_finding (the named CC '
                                 'image(s) carry a finding).')
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='none')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.inner = _Bilateral(
            data_root=opt.dataroot,
            annotations_csv=opt.annotations_csv,
            split=opt.split,
            img_size=tuple(opt.bilateral_size),
            flip_right=opt.flip_right,
            crop_width=opt.crop_width,
            finding_filter=opt.finding_filter,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        a, b = self.inner[index]
        l_path, r_path = self.inner.pairs[index]
        item = {'A': a, 'B': b, 'A_paths': str(l_path), 'B_paths': str(r_path)}
        ma = _load_mask(_mask_path(l_path), self.inner.img_size, False, self.inner.crop_width)
        mb = _load_mask(_mask_path(r_path), self.inner.img_size, self.inner.flip_right, self.inner.crop_width)
        if ma is not None and mb is not None:
            item['A_mask'], item['B_mask'] = ma, mb
        return item
