import os
import pandas as pd
from base.base_dataset import MCQVideoDataset


class WebVid(MCQVideoDataset):

    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def _load_metadata(self):
        metadata_dir = './meta_data'
        split_files = {            
            'train': 'webvid_training_success_full_noun_verb.tsv',
            'val': 'webvid_validation_success_full_noun_verb.tsv',
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata

        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary


    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, self.split, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[0]
