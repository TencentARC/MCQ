from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler


class MultiDistBaseDataLoaderExplicitSplit(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, args, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.args = args
        self.train_sampler = DistributedSampler(dataset, num_replicas=self.args.world_size,
                                                rank=self.args.rank, drop_last=True)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True,
            'sampler': self.train_sampler
        }
        super().__init__(**self.init_kwargs)


class BaseDataLoaderExplicitSplit(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)
        