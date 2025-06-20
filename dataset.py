import numpy as np
import json
import torch

from torch.utils.data import Dataset


class AnomalyDetectionDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.inputs = []
        self.labels = []
        self.setup_dataset()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = np.load(f'{self.root}/{self.inputs[index]}')
        label = np.load(f'{self.root}/{self.labels[index]}')

        input = torch.from_numpy(input)
        label = torch.from_numpy(label)

        return input.float(), label.float(), label.shape[0]

    def setup_dataset(self):
        annotations = self.load_annotations()

        for annotation in annotations:
            annotation = json.loads(annotation.strip())

            input = annotation['input']
            label = annotation['label']

            self.inputs.append(input)
            self.labels.append(label)

    def load_annotations(self):
        with open(f'{self.root}/annotations.json', 'r') as annotations:
            return annotations.readlines()
