import numpy as np
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from util import dist as udist


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices=('CUT', 'cut', 'FastCUT', 'fastcut'))

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--bidirectional',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Train a single shared G symmetrically: fake_R=G(real_A) AND fake_L=G(real_B). "
                                 "Applies the adversarial loss and the standard PatchNCE in both directions. "
                                 "Intended for the contralateral (L-CC <-> R-CC) setup; requires --flip_right so "
                                 "L and R share one canonical domain. No-ops for stock CUT/FastCUT when off.")
        parser.add_argument('--masked_loss',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Restrict NCE patches and the GAN loss to the foreground mask, and zero the background of the G/D inputs. Requires A_mask/B_mask from the dataset; no-ops when masks are absent.")
        parser.add_argument('--mask_bg_value', type=float, default=-1.0,
                            help="Value used to fill the masked-out background (default -1.0 = black in the [-1, 1] range).")
        parser.add_argument('--mask_feather', type=int, default=3,
                            help="Gaussian feather radius (px) for blending foreground over the background, so masking introduces no hard step edge at the boundary. 0 = hard binary mask (old behaviour).")
        parser.add_argument('--mask_erode', type=int, default=1,
                            help="Erode the foreground by this many cells for the GAN/NCE loss masks, so the scored region stays strictly inside the kept (feathered) region and no boundary ring is left unconstrained. 0 = no erosion.")
        parser.add_argument('--lambda_L1', type=float, default=0.0,
                            help="weight for paired L1 reconstruction L1(G(A), B). Requires "
                                 "--dataset_mode bilateral (true paired R). 0 disables.")
        parser.add_argument('--lambda_L2', type=float, default=0.0,
                            help="weight for paired L2/MSE reconstruction MSE(G(A), B). Requires "
                                 "--dataset_mode bilateral (true paired R). 0 disables.")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        if opt.bidirectional:
            # flip_equivariance randomly H-flips the input, which undoes the canonical
            # orientation that --flip_right + crop_width (keep chest-wall columns)
            # establish for the contralateral setup. Override the (FastCUT) default so
            # the shared-G map stays oriented. An explicit --flip_equivariance True is
            # caught again in __init__.
            parser.set_defaults(flip_equivariance=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if opt.bidirectional and opt.flip_equivariance:
            print("[CUT] --bidirectional is set: disabling --flip_equivariance "
                  "(it H-flips inputs and undoes the --flip_right / crop_width canonical orientation).")
            opt.flip_equivariance = False

        # Paired pixel reconstruction (L1/L2) is only meaningful when real_B is the
        # *true* contralateral of real_A — i.e. the paired `bilateral` adapter.
        # unpaired_bilateral / scheduled_bilateral hand out a random R, so a pixel
        # target is undefined there; fail loudly rather than train on noise.
        self.use_recon = self.isTrain and (opt.lambda_L1 > 0.0 or opt.lambda_L2 > 0.0)
        if self.use_recon and opt.dataset_mode != 'bilateral':
            raise ValueError(
                f"--lambda_L1/--lambda_L2 need paired targets: use --dataset_mode bilateral "
                f"(got {opt.dataset_mode!r}). unpaired_bilateral/scheduled_bilateral provide a "
                "random R, so pixel reconstruction is undefined.")

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if opt.bidirectional and self.isTrain:
            # Second direction R->L: G(real_B) is a real translation (fake_L), scored
            # by its own GAN + standard NCE (NCE_Y). Add the log/visual if not already
            # present from nce_idt.
            if 'NCE_Y' not in self.loss_names:
                self.loss_names += ['NCE_Y']
            self.visual_names += ['fake_L']

        if self.use_recon:
            if opt.lambda_L1 > 0.0:
                self.loss_names += ['recon_L1']
            if opt.lambda_L2 > 0.0:
                self.loss_names += ['recon_L2']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)  # also reused for the L1 recon term
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        # Under DDP each rank already receives only its shard, so the local
        # data["A"] *is* the per-GPU batch — don't divide further.
        if udist.is_ddp():
            bs_per_gpu = data["A"].size(0)
        else:
            bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        if self.mask_A is not None:
            self.mask_A = self.mask_A[:bs_per_gpu]
        if self.mask_B is not None:
            self.mask_B = self.mask_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # Optional foreground masks, aligned with real_A / real_B respectively.
        mask_a = input.get('A_mask' if AtoB else 'B_mask')
        mask_b = input.get('B_mask' if AtoB else 'A_mask')
        self.mask_A = mask_a.to(self.device) if mask_a is not None else None
        self.mask_B = mask_b.to(self.device) if mask_b is not None else None
        self.use_mask = bool(self.opt.masked_loss) and self.mask_A is not None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Run the shared G over both reals when we need the second direction: the
        # identity branch (nce_idt) OR the bidirectional shared-G branch. fake[:n] is
        # G(real_A)=fake_R, fake[n:] is G(real_B)=fake_L/idt_B.
        both = (self.opt.nce_idt or self.opt.bidirectional) and self.opt.isTrain
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if both else self.real_A
        # Foreground mask aligned with self.real (same cat / flip as the images).
        if self.use_mask:
            self.mask = torch.cat((self.mask_A, self.mask_B), dim=0) if both else self.mask_A
        else:
            self.mask = None
        if self.opt.flip_equivariance:
            # Under DDP every rank must agree on whether this step is flipped,
            # otherwise the NCE pairing diverges across ranks.
            local_flip = self.opt.isTrain and (np.random.random() < 0.5)
            self.flipped_for_equivariance = udist.broadcast_bool(local_flip, src=0, device=self.device)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                if self.mask is not None:
                    self.mask = torch.flip(self.mask, [3])

        if self.mask is not None:
            self.real = self._apply_mask(self.real, self.mask)  # zero background of G's input
        self.fake = self.netG(self.real)
        if self.mask is not None:
            self.fake = self._apply_mask(self.fake, self.mask)  # zero background of G's output
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        if both and self.opt.bidirectional:
            # Second direction: fake_L = G(real_B). Same tensor as idt_B when nce_idt
            # is also on, but named for its translation (not identity) role.
            self.fake_L = self.fake[self.real_A.size(0):]

    def _feather(self, mask):
        """Gaussian-blur a binary mask into a soft alpha in [0, 1].

        Blending with a feathered alpha (instead of a hard 0/1 multiply) removes
        the step edge at the mask boundary that a convolutional G + PatchGAN
        otherwise reproduce as a ghost contour. ``--mask_feather 0`` restores the
        old hard mask. A pixel whose whole ``(2r+1)`` neighbourhood is background
        gets alpha exactly 0, so the deep background stays at ``mask_bg_value``.
        """
        r = int(self.opt.mask_feather)
        if r <= 0:
            return mask
        k = getattr(self, '_feather_kernel', None)
        if k is None or k.shape[-1] != 2 * r + 1:
            ax = torch.arange(2 * r + 1, dtype=torch.float32) - r
            g = torch.exp(-(ax ** 2) / (2 * (r / 2.0) ** 2))
            g = g / g.sum()
            k = (g[:, None] * g[None, :]).view(1, 1, 2 * r + 1, 2 * r + 1)
            self._feather_kernel = k
        k = k.to(mask.device, mask.dtype)
        # Replicate-pad (not zero-pad) so tissue reaching the image border — e.g.
        # the chest-wall crop edge — keeps alpha ~= 1 instead of being feathered
        # toward the background and creating a new edge there.
        return F.conv2d(F.pad(mask, (r, r, r, r), mode='replicate'), k)

    def _apply_mask(self, x, mask):
        """Blend foreground over a constant background via a feathered alpha.

        ``x * a + mask_bg_value * (1 - a)`` with ``a = feather(mask)``; far from
        the boundary this keeps foreground pixels and fills background with
        ``mask_bg_value`` exactly, but the transition is a smooth ramp.
        """
        a = self._feather(mask)
        return x * a + self.opt.mask_bg_value * (1.0 - a)

    def _recon(self, fake, real, mask_src):
        """(L1, L2) reconstruction between a translation and its paired target.

        ``fake`` is already mask-blended in ``forward()``; mask ``real`` the same
        way so the constant background cancels and only the foreground is scored.
        Each term is computed only when its weight is positive (else a zero scalar).
        """
        if mask_src is not None:
            real = self._apply_mask(real, mask_src)
        l1 = self.criterionIdt(fake, real) if self.opt.lambda_L1 > 0.0 else fake.new_zeros(())
        l2 = self.criterionL2(fake, real) if self.opt.lambda_L2 > 0.0 else fake.new_zeros(())
        return l1, l2

    @staticmethod
    def _erode(mask, r):
        """Binary morphological erosion by radius ``r`` cells (min-pool). No-op for r<=0."""
        if r <= 0:
            return mask
        k = 2 * r + 1
        return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=r)

    def _gan_mask(self, mask, pred):
        """Downsample a foreground mask to the discriminator output grid, eroded.

        Area interpolation + re-binarize marks a PatchGAN cell as foreground when
        the majority of its footprint is foreground; eroding by ``--mask_erode``
        cells then drops the boundary ring so the scored region sits strictly
        inside the kept (feathered) region — otherwise those straddling cells are
        unconstrained and G is free to grow a spurious edge there. An image whose
        mask erodes to nothing falls back to its un-eroded cells.
        """
        m = (F.interpolate(mask, size=pred.shape[-2:], mode='area') > 0.5).float()
        er = self._erode(m, self.opt.mask_erode)
        if self.opt.mask_erode > 0:
            empty = er.flatten(1).sum(1) == 0
            if bool(empty.any()):
                er[empty] = m[empty]
        return er

    def _D_direction(self, fake, fake_mask_src, real_img, real_mask_src):
        """One adversarial direction's discriminator terms.

        ``fake`` is scored as fake (gradient to G stopped by detaching), ``real_img``
        as real. ``fake_mask_src`` / ``real_mask_src`` are the full-res foreground
        masks aligned with ``fake`` / ``real_img``; ``_gan_mask`` downsamples + erodes
        them to the PatchGAN grid. Returns ``(loss_fake, loss_real, pred_real)``.
        """
        pred_fake = self.netD(fake.detach())
        fmask = self._gan_mask(fake_mask_src, pred_fake) if self.use_mask else None
        loss_fake = self.criterionGAN(pred_fake, False, mask=fmask).mean()
        real_in = self._apply_mask(real_img, real_mask_src) if self.use_mask else real_img
        pred_real = self.netD(real_in)
        rmask = self._gan_mask(real_mask_src, pred_real) if self.use_mask else None
        loss_real = self.criterionGAN(pred_real, True, mask=rmask).mean()
        return loss_fake, loss_real, pred_real

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator (both directions if --bidirectional)."""
        n = self.real_A.size(0)
        mask_R = self.mask[:n] if self.use_mask else None  # aligned with fake_B = G(L)
        self.loss_D_fake, self.loss_D_real, self.pred_real = self._D_direction(
            self.fake_B, mask_R, self.real_B, self.mask_B)

        if self.opt.bidirectional:
            mask_L = self.mask[n:] if self.use_mask else None  # aligned with fake_L = G(R)
            b_fake, b_real, _ = self._D_direction(self.fake_L, mask_L, self.real_A, self.mask_A)
            self.loss_D_fake = (self.loss_D_fake + b_fake) * 0.5
            self.loss_D_real = (self.loss_D_real + b_real) * 0.5

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def _G_GAN_direction(self, fake, fake_mask_src):
        """Generator-side GAN loss for one direction (fake should fool D)."""
        pred_fake = self.netD(fake)
        fmask = self._gan_mask(fake_mask_src, pred_fake) if self.use_mask else None
        return self.criterionGAN(pred_fake, True, mask=fmask).mean()

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator (both directions if --bidirectional)."""
        n = self.real_A.size(0)
        # First, G(A) should fake the discriminator (and G(B) too, if bidirectional)
        if self.opt.lambda_GAN > 0.0:
            g = self._G_GAN_direction(self.fake_B, self.mask[:n] if self.use_mask else None)
            if self.opt.bidirectional:
                g = (g + self._G_GAN_direction(self.fake_L, self.mask[n:] if self.use_mask else None)) * 0.5
            self.loss_G_GAN = g * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B, self.mask_A if self.use_mask else None)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        # Second-direction standard PatchNCE: the nce_idt identity term and the
        # bidirectional R->L structure term share the same form NCE(real_B, G(real_B)).
        if self.opt.lambda_NCE > 0.0 and (self.opt.nce_idt or self.opt.bidirectional):
            tgt = self.idt_B if self.opt.nce_idt else self.fake_L
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, tgt, self.mask_B if self.use_mask else None)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # Paired pixel reconstruction (only on the paired `bilateral` adapter; the
        # guard in __init__ enforces that). Symmetric under --bidirectional: the
        # R->L translation reconstructs real_A, averaged to keep the same budget.
        self.loss_recon_L1 = self.loss_recon_L2 = 0.0
        if self.use_recon:
            l1, l2 = self._recon(self.fake_B, self.real_B, self.mask_B if self.use_mask else None)
            if self.opt.bidirectional:
                l1b, l2b = self._recon(self.fake_L, self.real_A, self.mask_A if self.use_mask else None)
                l1, l2 = (l1 + l1b) * 0.5, (l2 + l2b) * 0.5
            self.loss_recon_L1 = self.opt.lambda_L1 * l1
            self.loss_recon_L2 = self.opt.lambda_L2 * l2

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_recon_L1 + self.loss_recon_L2
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt, mask=None):
        n_layers = len(self.nce_layers)
        if mask is not None:
            src = self._apply_mask(src, mask)  # encode the same masked source G saw
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None, mask=mask)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
