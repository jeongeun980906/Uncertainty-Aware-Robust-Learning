import os
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import torch
import numpy as np
from torchvision.datasets.mnist import MNIST, VisionDataset
from torchvision.datasets.utils import download_url, extract_archive, verify_str_arg
from torchvision.transforms import Compose, Normalize, ToTensor
from dataloader.utils import noisify

# Cell

MNIST_NORMALIZATION = Normalize((0.1307,), (0.3081,))

# Cell

# based on torchvision.datasets.mnist.py (https://github.com/pytorch/vision/blob/37eb37a836fbc2c26197dfaf76d2a3f4f39f15df/torchvision/datasets/mnist.py)
class AmbiguousMNIST(VisionDataset):
    """
    Ambiguous-MNIST Dataset
    Please cite:
        @article{mukhoti2021deterministic,
          title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
          author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
          journal={arXiv preprint arXiv:2102.11582},
          year={2021}
        }
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Normalize the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for max throughput)
    """

    mirrors = ["http://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v1.0.0/"]

    resources = dict(
        data=("amnist_samples.pt", "4f7865093b1d28e34019847fab917722"),
        targets=("amnist_labels.pt", "3bfc055a9f91a76d8d493e8b898c3c95"),
    )

    def __init__(
        self,
        root: str,
        *,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        normalize: bool = True,
        noise_stddev=0.05,
        device=None,noise_type='symmetric',test_noisy=True,
        noise_rate=0.2,num=2, indicies=None
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.noise_type=noise_type
        if download:
            self.download()

        self.data = torch.load(self.resource_path("data"), map_location='cpu')
        if normalize:
            self.data = self.data.sub_(0.1307).div_(0.3081)

        self.targets = torch.load(self.resource_path("targets"), map_location='cpu')

        # Each sample has `num_multi_labels` many labels.
        num_multi_labels = self.targets.shape[1]

        # Flatten the multi-label dataset into a single-label dataset with samples repeated x `num_multi_labels` many times
        self.data = self.data.expand(-1, num_multi_labels, 28, 28).reshape(-1,1, 28, 28)
        self.targets = self.targets.reshape(-1)

        if self.train:
            data_range = slice(None,60000)
        else:
            if test_noisy:
                data_range = slice(60000,120000,6)
            else:
                data_range = slice(60000,60000)
        # data_range = slice(None, 60000) if self.train slice(60000,None) elif test_noisy else slice(None, None)
        self.data = self.data[data_range]
        print('noisy',self.train,self.data.shape)
        if noise_stddev > 0.0:
            self.data += torch.randn_like(self.data) * noise_stddev

        self.targets = self.targets[data_range]
        if noise_type != 'clean':
            self.targets=np.asarray([[self.targets[i]] for i in range(len(self.targets))])
            self.noisy_labels, self.transition_matrix = noisify(dataset='mnist', train_labels=self.targets, noise_type=noise_type, noise_rate=noise_rate, num=num)
            self.noisy_labels=[i[0] for i in self.noisy_labels]
            _targets=[i[0] for i in self.targets]
            self.noise_or_not = np.transpose(self.noisy_labels)==np.transpose(_targets)
            self.data= self.data.to(device)
            self.noisy_labels = torch.tensor(self.noisy_labels, dtype=torch.int64).to(device)                
        else:
            self.data, self.targets = self.data.to(device), self.targets.to(device)
        try:
            if indicies.any() != None:
                indicies = torch.tensor(indicies,dtype=torch.int64).to(device)
                self.data = self.data[indicies]
                self.targets = self.targets[indicies]
        except:
            pass
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.noise_type != 'clean':
            img, target = self.data[index], self.noisy_labels[index]
        else:
            img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def data_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    def resource_path(self, name):
        return os.path.join(self.data_folder, self.resources[name][0])

    def _check_exists(self) -> bool:
        return all(os.path.exists(self.resource_path(name)) for name in self.resources)

    def download(self) -> None:
        """Download the data if it doesn't exist in data_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.data_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources.values():
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_url(url, root=self.data_folder, filename=filename, md5=md5)
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                except:
                    raise
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

        print("Done!")

# Cell


class FastMNIST(MNIST):
    """
    FastMNIST, based on https://tinyurl.com/pytorch-fast-mnist. It's like MNIST (<http://yann.lecun.com/exdb/mnist/>) but faster.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Normalize the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for
            max throughput).
    """

    def __init__(self, *args, normalize=True, noise_stddev=0.05, device,indicies=None,**kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        # self.noise_type=noise_type
        if normalize:
            self.data = self.data.sub_(0.1307).div_(0.3081)
        if noise_stddev > 0.0:
            self.data += torch.randn_like(self.data) * noise_stddev
        # if noise_type != 'clean':
        #     self.targets=np.asarray([[self.targets[i]] for i in range(len(self.targets))])
        #     self.noisy_labels, self.transition_matrix = noisify(dataset='mnist', train_labels=self.targets, noise_type=noise_type, noise_rate=noise_rate, num=num)
        #     self.noisy_labels=[i[0] for i in self.noisy_labels]
        #     _targets=[i[0] for i in self.targets]
        #     self.noise_or_not = np.transpose(self.noisy_labels)==np.transpose(_targets)
        #     self.data= self.data.to(device)
        #     self.noisy_labels = torch.tensor(self.noisy_labels, dtype=torch.int64).to(device)
        # else:
        self.data, self.targets = self.data.to(device), self.targets.to(device)
        print('clean',self.train,self.data.shape)
        try:
            if indicies.any() != None:
                indicies = torch.tensor(indicies,dtype=torch.int64).to(device)
                self.data = self.data[indicies]
                self.targets = self.targets[indicies]
        except:
            pass
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if self.noise_type != 'clean':
        #     img, target = self.data[index], self.noisy_labels[index]
        # else:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# Cell


def DirtyMNIST(
    root: str,
    *,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    normalize=True,
    noise_stddev=0.05,
    device=None,
    noise_type='symmetric',
    noise_rate=0,test_noisy=True,
    clean_indicies=None,ambiguous_indicies=None
):
    """
    DirtyMNIST
    Please cite:
        @article{mukhoti2021deterministic,
          title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
          author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
          journal={arXiv preprint arXiv:2102.11582},
          year={2021}
        }
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Normalize the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for
            max throughput).
    """

    mnist_dataset = FastMNIST(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        normalize=normalize,
        noise_stddev=noise_stddev,
        device=device,  indicies = clean_indicies
    )

    amnist_dataset = AmbiguousMNIST(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        normalize=normalize,
        noise_stddev=noise_stddev,test_noisy = test_noisy,
        device=device,noise_type=noise_type,noise_rate=noise_rate, indicies=ambiguous_indicies
    )

    return torch.utils.data.ConcatDataset([mnist_dataset, amnist_dataset])