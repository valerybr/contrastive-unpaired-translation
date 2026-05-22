import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def _is_main_process():
    # Under torchrun/DDP, RANK is set per-process. Default 0 covers single-process
    # (DataParallel or CPU/single-GPU) where everyone is "main".
    return int(os.environ.get("RANK", "0")) == 0


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.is_main = _is_main_process()
        if opt.display_id is None:
            self.display_id = np.random.randint(100000) * 10  # just a random display id
        else:
            self.display_id = opt.display_id
        # Only the main process drives display/HTML/wandb under DDP; other ranks no-op.
        self.use_html = opt.isTrain and not opt.no_html and self.is_main
        self.use_wandb = bool(getattr(opt, 'use_wandb', False)) and opt.isTrain and self.is_main
        self.wandb_run = None
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0 and self.is_main:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.plot_data = {}
            self.ncols = opt.display_ncols
            if "tensorboard_base_url" not in os.environ:
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            else:
                self.vis = visdom.Visdom(port=2004,
                                         base_url=os.environ['tensorboard_base_url'] + '/visdom')
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_wandb:
            import wandb
            import hashlib
            self.wandb = wandb
            # Stable run id derived from the run name so retries/--continue_train
            # land in the same wandb run instead of creating a new one each launch.
            run_name = opt.wandb_run_name or opt.name
            run_id = hashlib.sha1(run_name.encode('utf-8')).hexdigest()[:16]
            self.wandb_run = wandb.init(
                project=opt.wandb_project,
                entity=opt.wandb_entity,
                id=run_id,
                name=run_name,
                config=vars(opt),
                dir=os.path.join(opt.checkpoints_dir, opt.name),
                resume='allow',
            )
            # Plot every metric against epoch (fractional within an epoch) rather
            # than wandb's internal step counter, which otherwise tracks the
            # number of images seen. Each log call carries an 'epoch' value.
            self.wandb.define_metric('epoch')
            self.wandb.define_metric('*', step_metric='epoch')

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        if self.is_main:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if not self.is_main:
            return
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, ncols, 2, self.display_id + 1,
                                    None, dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(
                            image_numpy.transpose([2, 0, 1]),
                            self.display_id + idx,
                            None,
                            dict(title=label)
                        )
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

        if self.use_wandb:
            log_payload = {'epoch': epoch}
            for label, image in visuals.items():
                log_payload['images/%s' % label] = self.wandb.Image(util.tensor2im(image), caption=label)
            self.wandb.log(log_payload)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if len(losses) == 0 or not self.is_main:
            return

        if self.display_id > 0:
            plot_name = '_'.join(list(losses.keys()))

            if plot_name not in self.plot_data:
                self.plot_data[plot_name] = {'X': [], 'Y': [], 'legend': list(losses.keys())}

            plot_data = self.plot_data[plot_name]
            plot_id = list(self.plot_data.keys()).index(plot_name)

            plot_data['X'].append(epoch + counter_ratio)
            plot_data['Y'].append([losses[k] for k in plot_data['legend']])
            try:
                self.vis.line(
                    X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                    Y=np.array(plot_data['Y']),
                    opts={
                        'title': self.name,
                        'legend': plot_data['legend'],
                        'xlabel': 'epoch',
                        'ylabel': 'loss'},
                    win=self.display_id - plot_id)
            except VisdomExceptionBase:
                self.create_visdom_connections()

        if self.use_wandb:
            payload = {'train/%s' % k: v for k, v in losses.items()}
            payload['epoch'] = epoch + counter_ratio
            self.wandb.log(payload)

    def log_epoch_averages(self, epoch, avg_losses, lr=None):
        """Log per-epoch averaged losses (and optionally LR) to wandb."""
        if not self.use_wandb:
            return
        payload = {'epoch_avg/%s' % k: v for k, v in avg_losses.items()}
        payload['epoch'] = epoch
        if lr is not None:
            payload['train/lr'] = lr
        self.wandb.log(payload)

    def watch_model(self, model):
        """Optional: track gradients/parameters of the GAN networks."""
        if not self.use_wandb or not getattr(self.opt, 'wandb_watch_model', False):
            return
        for name in getattr(model, 'model_names', []):
            net = getattr(model, 'net' + name, None)
            if net is not None:
                self.wandb.watch(net, log='gradients', log_freq=max(self.opt.print_freq, 1) * 100, idx=hash(name) & 0xFFFF)

    def finish(self):
        if self.use_wandb and self.wandb_run is not None:
            self.wandb.finish()
            self.wandb_run = None

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if not self.is_main:
            return
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
