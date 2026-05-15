import os
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util import dist as udist


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    if udist.is_main():
        print('[ddp] world_size=%d visible_gpus=%d CUDA_VISIBLE_DEVICES=%r local_rank=%d backend=%s' % (
            udist.get_world_size(),
            torch.cuda.device_count(),
            os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            udist.get_local_rank(),
            ('nccl' if torch.cuda.is_available() else 'cpu/gloo'),
        ))

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    if udist.is_main():
        print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    # On --continue_train, align the wandb x-axis with global progress by
    # reconstructing total_iters from the resumed epoch. Per-rank total_iters
    # advances over each rank's DistributedSampler shard, so divide by world
    # size. Assumes the previous run stopped at an epoch boundary (the usual
    # case via save_epoch_freq).
    total_iters = (opt.epoch_count - 1) * dataset_size // udist.get_world_size() if opt.continue_train else 0

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        epoch_loss_sums = {}
        epoch_loss_count = 0
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                # Phase 1: lazy-build netF on every rank using the local batch.
                # PatchSampleF.create_mlp materializes Linear layers from the
                # first forward pass's feature shapes — has to happen on every
                # rank BEFORE DDP wraps netF (DDP snapshots the parameter set
                # at construction, so any params added later won't sync).
                model.data_dependent_initialize(data)
                # Phase 2: regular setup — build schedulers, load checkpoints
                # if --continue_train, etc.
                model.setup(opt)
                # Phase 3: broadcast every net's weights from rank 0 so all
                # ranks start from identical parameters (per-rank RNG can
                # diverge during init, and lazy MLPs are initialized with the
                # current RNG state).
                if udist.is_ddp():
                    udist.barrier()
                    for name in model.model_names:
                        net = getattr(model, 'net' + name)
                        udist.broadcast_module(net, src=0)
                # Phase 4: wrap each net in DDP (or DataParallel for legacy).
                model.parallelize()
                visualizer.watch_model(model)
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            for k, v in model.get_current_losses().items():
                epoch_loss_sums[k] = epoch_loss_sums.get(k, 0.0) + float(v)
            epoch_loss_count += 1

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, step=total_iters)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0 or getattr(opt, 'use_wandb', False):
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses, step=total_iters)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                if udist.is_main():
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)  # internally rank-0-gated
                udist.barrier()  # wait for rank 0's I/O so others don't get ahead and timeout

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            if udist.is_main():
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            udist.barrier()  # resync after rank-0-only I/O

        avg_losses = {}
        if epoch_loss_count > 0:
            avg_losses = {k: total / epoch_loss_count for k, total in epoch_loss_sums.items()}
            if udist.is_main():
                avg_msg = '(epoch %d avg over %d iters) ' % (epoch, epoch_loss_count)
                for k, v in avg_losses.items():
                    avg_msg += '%s: %.3f ' % (k, v)
                print(avg_msg)
                with open(visualizer.log_name, "a") as log_file:
                    log_file.write('%s\n' % avg_msg)

        if udist.is_main():
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        lr = model.update_learning_rate()                # update learning rates at the end of every epoch.
        visualizer.log_epoch_averages(epoch, avg_losses, lr=lr, step=total_iters)

    visualizer.finish()
    udist.cleanup()
