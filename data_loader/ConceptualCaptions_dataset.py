import os
import pandas as pd
from base.base_dataset import MCQImageDataset


class ConceptualCaptions3M(MCQImageDataset):
    """
    Conceptual Captions dataset. Split files are specific to my download regime.
    """

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data'
        split_files = {
            'train': 'cc3m_training_success_full_noun_verb.tsv',
            'val': 'cc3m_validation_success_full_noun_verb.tsv',            # there is no test
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata

        self.metadata = metadata

    def _get_video_path(self, sample):
        # conceptual captions uses this hashing to create the filename
        rel_dir = 'training'
        if self.split != 'train':
            rel_dir = 'validation'
        rel_fp = os.path.join(rel_dir, sample[1])
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]
