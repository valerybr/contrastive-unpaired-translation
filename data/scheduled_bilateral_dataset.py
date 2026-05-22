"""CUT-compatible adapter around `data.bilateral.ScheduledBilateralDataset`."""

from data.base_dataset import BaseDataset
from data.bilateral import ScheduledBilateralDataset as _Scheduled


def _parse_schedule(s: str) -> list[tuple[int, float]]:
    segments: list[tuple[int, float]] = []
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        epochs_str, p_str = chunk.split(':')
        segments.append((int(epochs_str), float(p_str)))
    return segments


class ScheduledBilateralDataset(BaseDataset):
    """Wraps the scheduled L CC / R CC mammography dataset for CUT training.

    A → left CC, B → either the true paired right CC or a randomly sampled
    right CC, with the random-pair probability driven by a per-epoch schedule
    (e.g. ``20:0.5,20:0.3,20:0.1,20:0.0``).
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
        parser.add_argument('--pair_schedule', type=str, default='',
                            help='Random-pair probability schedule as '
                                 '"epochs:p,epochs:p,..." e.g. '
                                 '"20:0.5,20:0.3,20:0.1,20:0.0". '
                                 'Empty = constant p=0.0 (always paired). '
                                 'Past the end of the schedule, p holds at '
                                 'the last value.')
        parser.add_argument('--pair_schedule_seed', type=int, default=0,
                            help='Base seed for per-sample pair RNG')
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='none')
        return parser

    def __init__(self, opt):
        self.schedule = _parse_schedule(opt.pair_schedule)
        self.inner = _Scheduled(
            data_root=opt.dataroot,
            annotations_csv=opt.annotations_csv,
            split=opt.split,
            img_size=tuple(opt.bilateral_size),
            flip_right=opt.flip_right,
            seed=opt.pair_schedule_seed,
            crop_width=opt.crop_width,
        )
        BaseDataset.__init__(self, opt)

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        self._current_epoch = int(value)
        self.inner.set_epoch_state(self._current_epoch, self._p_for_epoch(self._current_epoch))

    def _p_for_epoch(self, epoch: int) -> float:
        if not self.schedule:
            return 0.0
        cum = 0
        for n, p in self.schedule:
            cum += n
            if epoch <= cum:
                return p
        return self.schedule[-1][1]

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        # r_path is the actually-resolved right image (the random pick under the
        # schedule, not necessarily the true pair), so B_paths labels B correctly.
        a, b, l_path, r_path = self.inner[index]
        return {'A': a, 'B': b, 'A_paths': str(l_path), 'B_paths': str(r_path)}
