import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from pacs import PACS
from PIL import Image
import pdb
from torchvision import transforms
from PIL import ImageFilter
import random
from visdac import VISDAC
from domainnet import DomainNet

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

class dataset(Dataset):
    def __init__(self, dataset, root, mode, transform, noisy_path=None, sens_classes=5):
        self.dataset = dataset
        self.root = root
        self.mode = mode
        self.transform = transform
        self.noisy_path = noisy_path
        self.sens_classes = sens_classes
        self.parse_dataset()

    def parse_dataset(self):
        if self.dataset == 'mnist':
            return self.get_mnist()
        elif self.dataset == 'mnistm':
            return self.get_mnistm()
        elif self.dataset == 'svhn':
            return self.get_svhn()
        elif self.dataset == 'usps':
            return self.get_usps()
        elif self.dataset.split('/')[0] == 'pacs':
            return self.get_pacs()
        elif self.dataset.split('/')[0] == 'visdac':
            return self.get_visdac()
        elif self.dataset.split('/')[0] == 'domainnet':
            return self.get_domainnet()
        elif 'domainnet' in self.dataset.split('/')[0]:
            return self.get_domainnet()

    def get_pacs(self):
        domain = self.dataset.split('/')[-1]

        if self.mode == 'all':
            train_set = PACS(root=self.root,
                                    domain=domain,
                                    train=True,
                                    transform=self.transform,
                                    from_file=False
                                    )
            
            test_set = PACS(root=self.root,
                                    domain=domain,
                                    train=False,
                                    transform=self.transform,
                                    from_file=False
                                    )
            
            data = np.concatenate((train_set.data, test_set.data))
            labels = np.concatenate((train_set.labels, test_set.labels))

        else:
            train = True if self.mode == 'train' else False
            dataset = PACS(root=self.root,
                                    domain=domain,
                                    train=train,
                                    transform=self.transform,
                                    from_file=False
                                    )
            data = dataset.data
            labels = dataset.labels

        self.strong_augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        if self.noisy_path is not None:
            idx, noisy_labels = self.load_noisy_labels()

            self.data = data[idx]
            self.labels = labels[idx]
            self.noisy_labels = noisy_labels
        else:
            self.data = data
            self.labels = labels

    def get_visdac(self):
        domain = self.dataset.split('/')[-1]

        if self.mode == 'all':
            train_set = VISDAC(root=self.root,
                                    domain=domain,
                                    train=True,
                                    transform=self.transform,
                                    from_file=False
                                    )
            
            test_set = VISDAC(root=self.root,
                                    domain=domain,
                                    train=False,
                                    transform=self.transform,
                                    from_file=False
                                    )
            
            data = np.concatenate((train_set.data, test_set.data))
            labels = np.concatenate((train_set.labels, test_set.labels))

        else:
            train = True if self.mode == 'train' else False
            dataset = VISDAC(root=self.root,
                                    domain=domain,
                                    train=train,
                                    transform=self.transform,
                                    from_file=False
                                    )
            data = dataset.data
            labels = dataset.labels

        self.strong_augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.data = data
        self.labels = labels

    def get_domainnet(self):
        domain = self.dataset.split('/')[-1]

        if self.mode == 'all':
            train_set = DomainNet(root=self.root,
                                    domain=domain,
                                    train=True,
                                    transform=self.transform,
                                    from_file=False
                                    )
            
            test_set = DomainNet(root=self.root,
                                    domain=domain,
                                    train=False,
                                    transform=self.transform,
                                    from_file=False
                                    )
            
            data = np.concatenate((train_set.data, test_set.data))
            labels = np.concatenate((train_set.labels, test_set.labels))
            sensitives = np.concatenate((train_set.sensitives, test_set.sensitives)) 

        else:
            train = True if self.mode == 'train' else False
            dataset = DomainNet(root=self.root,
                                    domain=domain,
                                    train=train,
                                    transform=self.transform,
                                    from_file=False
                                    )
            data = dataset.data
            labels = dataset.labels
            sensitives = dataset.sensitives

        self.strong_augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.data = data
        self.labels = labels
        self.sensitives = sensitives

    def load_noisy_labels(self):
        idx = np.load(self.noisy_path+"_idx.npy")
        labels = np.load(self.noisy_path+"_noisylab.npy")

        return idx, labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, sensitive = self.data[index], self.labels[index], self.sensitives[index]
        noisy_target = self.noisy_labels[index] if self.noisy_path is not None else self.labels[index]

        img = Image.open(img)

        strong_augmented = self.strong_augmentation(img)
        strong_augmented2 = self.strong_augmentation(img)
        weak_augmented = self.transform(img) if self.transform is not None else img

        return weak_augmented, strong_augmented, target, index, noisy_target, strong_augmented2, sensitive

    def __len__(self) -> int:
        return len(self.data)

    def group_counts(self, resample_which, flag=''):
        if resample_which == 'group' or resample_which == 'balanced' or resample_which == 'natural':
            group_array = self.sensitives.tolist()
            if resample_which == 'balanced' or resample_which == 'natural':
                labels = self.labels.tolist()
                num_labels = len(set(labels))
                num_groups = len(set(group_array))
                group_array = (np.asarray(group_array) * num_labels + np.asarray(labels)).tolist()
        elif resample_which == 'class':
            group_array = self.labels.tolist()
            num_labels = len(set(group_array))

        self._group_array = torch.LongTensor(group_array)
        if resample_which == 'group':
            self._group_counts = (torch.arange(self.sens_classes).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'balanced':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'class':
            self._group_counts = (torch.arange(num_labels).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'natural':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
            self._group_counts = torch.ones_like(self._group_counts)

        if 'Younger' in flag and resample_which in ['group', 'balanced', 'natural']:
            self._group_counts[-1] *= 10
            self._group_counts[-2] *= 10
            if resample_which == 'balanced' or resample_which == 'natural':
                self._group_counts[-3] *= 10
                self._group_counts[-4] *= 10
        if 'Older' in flag and resample_which in ['group', 'balanced', 'natural']:
            self._group_counts[-1] /= 10
            self._group_counts[-2] /= 10
            if resample_which == 'balanced' or resample_which == 'natural':
                self._group_counts[-3] /= 10
                self._group_counts[-4] /= 10

        return group_array, self._group_counts

    def get_weights(self, resample_which, flag=''):
        sens_attr, group_num = self.group_counts(resample_which, flag)
        group_weights = [1/max(1, x.item()) for x in group_num]
        sample_weights = [group_weights[int(i)] for i in sens_attr]
        return sample_weights
