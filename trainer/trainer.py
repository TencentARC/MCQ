import numpy as np
import torch
import torch.distributed as dist
from base import Multi_BaseTrainer_dist
from utils.util import inf_loop
from model.model_dist_MCQ import sim_matrix


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


class Trainer_MCQ(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.lr = config['optimizer']['args']['lr']
        self.schedule = config['trainer']['schedule']

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, lr, schedule):
        for milestone in schedule:
            if epoch >= milestone:
                lr = lr * 0.1
            else:
                lr = lr * 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def process_text(self, text_data):
        text_data = self.tokenizer(text_data, return_tensors='pt', padding=True,
                                      truncation=True)
        text_data = {key: val.to(self.device) for key, val in text_data.items()}
        return text_data

    def gather_tensor(self, embed):
        embed_all = [torch.zeros_like(embed) for _ in range(self.n_gpu)]
        torch.distributed.all_gather(embed_all, embed)
        embed_all = torch.cat(embed_all, dim=0)
        return embed_all

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                data['text'] = self.process_text(data['text'])
                data['question'] = self.process_text(data['question'])
                data['answer'] = self.process_text(data['answer'])
                data['video'] = data['video'].to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    text_cls_embeds, answer_cls_embeds, bridge_cls_embeds, video_cls_embeds = self.model(data)
                    text_cls_embeds = self.allgather(text_cls_embeds, self.n_gpu, self.args)
                    answer_cls_embeds = self.allgather(answer_cls_embeds, self.n_gpu, self.args)
                    bridge_cls_embeds = self.allgather(bridge_cls_embeds, self.n_gpu, self.args)
                    video_cls_embeds = self.allgather(video_cls_embeds, self.n_gpu, self.args)

                    output1 = sim_matrix(text_cls_embeds, video_cls_embeds)
                    output2 = sim_matrix(answer_cls_embeds, bridge_cls_embeds)

                    loss1 = self.loss(output1)
                    loss2 = self.loss(output2)
                    loss = loss1 + loss2

                loss.backward()
                self.optimizer.step()
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug('Train Epoch: {} dl{} {} Loss: {:.6f} Loss_vt: {:.6f} Loss_MCQ: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item(),
                        loss1.detach().item(),
                        loss2.detach().item()))

                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)
        
        self._adjust_learning_rate(self.optimizer, epoch, self.lr, self.schedule)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        answer_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        bridge_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))} 

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for _, data in enumerate(dl):
                    meta_arr[dl_idx].append(data['meta'])
                    data['text'] = self.process_text(data['text'])
                    data['question'] = self.process_text(data['question'])
                    data['answer'] = self.process_text(data['answer'])
                    data['video'] = data['video'].to(self.device)

                    text_embed, answer_embed, bridge_embed, vid_embed = self.model(data)

                    text_embed_all = self.gather_tensor(text_embed)
                    answer_embed_all = self.gather_tensor(answer_embed)
                    bridge_embed_all = self.gather_tensor(bridge_embed)
                    vid_embed_all = self.gather_tensor(vid_embed)

                    text_embed_arr[dl_idx].append(text_embed_all.cpu())
                    answer_embed_arr[dl_idx].append(answer_embed_all.cpu())
                    bridge_embed_arr[dl_idx].append(bridge_embed_all.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed_all.cpu())

                    sims_batch1 = sim_matrix(text_embed_all, vid_embed_all)
                    sims_batch2 = sim_matrix(answer_embed_all, bridge_embed_all)
                    loss = self.loss(sims_batch1) + self.loss(sims_batch2)
                    total_val_loss[dl_idx] += loss.item()

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None and self.args.local_rank == 0:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            answer_embeds = torch.cat(answer_embed_arr[dl_idx])
            bridge_embeds = torch.cat(bridge_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])

            sims_vt = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()
            sims_MCQ = sim_matrix(bridge_embeds, answer_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res_vt = metric(sims_vt)
                res_MCQ = metric(sims_MCQ)

                if self.args.rank == 0:
                    verbose(epoch=epoch, metrics=res_vt, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)
                    verbose(epoch=epoch, metrics=res_MCQ, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)

                nested_metrics[dl_idx][metric_name] = res_vt

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res_vt, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                        for dl_idx in range(len(self.valid_data_loader))}
            res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
