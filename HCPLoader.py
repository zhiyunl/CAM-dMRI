from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets

from utils import *


# prepare data


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def splitData(dataset, ratio):
    indices = list(range(len(dataset)))
    split_pos = int(np.floor(ratio * len(dataset)))
    np.random.shuffle(indices)  # incorporates shuffle here
    trn_idx, tst_idx = indices[split_pos:], indices[:split_pos]
    return trn_idx, tst_idx


def createLoader(idx, dataset, batch, shuff):
    sampler = SubsetRandomSampler(idx) if shuff else SequentialSampler(idx)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch)
    return loader


def split_trn_tst(train_data, valid_data, ratio=.2, batch_size=16):
    trn_idx, tst_idx = splitData(train_data, ratio)

    trn_sampler = SubsetRandomSampler(trn_idx)
    tst_sampler = SubsetRandomSampler(tst_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=trn_sampler, batch_size=batch_size, num_workers=2)
    validloader = torch.utils.data.DataLoader(valid_data, sampler=tst_sampler, batch_size=batch_size, num_workers=2,
                                              drop_last=True)

    return trainloader, validloader


def test_loader(source, trans_test):
    test_data = datasets.ImageFolder(source + "test/", transform=trans_test)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    print("test batch count: ", len(testloader))
    return testloader


def train_valid_loader(source, trans_train, trans_test):
    train_data = datasets.ImageFolder(source + "train/", transform=trans_train)
    valid_data = datasets.ImageFolder(source + "train/", transform=trans_test)

    trainloader, validloader = split_trn_tst(train_data, valid_data, batch_size=BATCH_SIZE, ratio=RATIO)
    print("train batch count: ", len(trainloader))
    print("validation batch count: ", len(validloader))
    return trainloader, validloader


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index])

    def __len__(self):
        return len(self.dataset)
