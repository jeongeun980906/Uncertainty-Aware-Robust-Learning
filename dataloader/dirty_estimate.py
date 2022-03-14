from dataloader.dirty_cifar import dirtyCIFAR10
from dataloader.dirty_mnist import DirtyMNIST
import torch
from torchvision import datasets,transforms

def get_estimated_dataset(indices_amb1,indices_clean1,indices_amb2,indices_clean2,args):
    if args.data == 'dirty_cifar10':
        transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
            ])
        clean_e_test = dirtyCIFAR10("./data", train=False, download=True,transform=transform_test,
                                            noise_type=args.mode,noise_rate=args.ER,mix_type='cutmix',
                                            alpha=args.alpha,clean_indicies=indices_clean1,ambiguous_indicies=indices_amb1)

        amb_e_test = dirtyCIFAR10("./data", train=False, download=True,transform=transform_test,
                                            noise_type=args.mode,noise_rate=args.ER,mix_type='cutmix',
                                            alpha=args.alpha,clean_indicies=indices_clean2,ambiguous_indicies=indices_amb2)
    elif args.data == 'dirty_mnist':
        clean_e_test = DirtyMNIST("./data", train=False, download=True, device="cpu",noise_type=args.mode,noise_rate=args.ER,
                                            clean_indicies=indices_clean1,ambiguous_indicies=indices_amb1)

        amb_e_test = DirtyMNIST("./data", train=False, download=True, device="cpu",noise_type=args.mode,noise_rate=args.ER,
                                            clean_indicies=indices_clean2,ambiguous_indicies=indices_amb2)
    ambiguous_e_dataloader = torch.utils.data.DataLoader(
        amb_e_test,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    clean_e_dataloader = torch.utils.data.DataLoader(
        clean_e_test,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return ambiguous_e_dataloader,clean_e_dataloader