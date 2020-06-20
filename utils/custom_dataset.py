import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, X, y, transform=None):
        '''
        Parameters
        ----------
        imgs : iterable of image paths.
        transform : callable. Optional transformer for samples.
        '''

        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        # expand dimensions if necessary

        sample = {'X': sample, 'y': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class CustomDataset2(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, X, y,z, transform=None):
        '''
        Parameters
        ----------
        imgs : iterable of image paths.
        transform : callable. Optional transformer for samples.
        '''

        self.X = X
        self.y = y
        self.names= z
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        name = self.names[idx]
        # expand dimensions if necessary

        sample = {'X': sample, 'y': label,'name':name}
        if self.transform:
            sample = self.transform(sample)

        return sample


'''
class RescaleUnit(object):
    Rescales images to unit range [0,1]

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = image - image.min()  # set min = 0
        image = image / image.max()  # max / max = 1
        sample = {'image': image, 'label': label}
        return sample
'''


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors'''

    def __init__(self, type='float'):
        self.type = type

    def __call__(self, sample):
        image = sample['X']
        label = sample['y']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {'X': torch.from_numpy(image).float(), 'y': label}
