
from os.path import join

from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.data import Dataset

'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''

class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


def get_class_per_split(args):
    a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
    c = c - a - b
    common_classes = [i for i in range(a)]
    source_private_classes = [i + a for i in range(b)]
    target_private_classes = [i + a + b for i in range(c)]

    source_classes = common_classes + source_private_classes
    target_classes = common_classes + target_private_classes

    return source_classes, target_classes, common_classes, source_private_classes, target_private_classes

def get_dataset_file(args):
    dataset = None
    if args.data.dataset.name == 'office':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['amazon', 'dslr', 'webcam'],
        files=[
            'amazon_reorgnized.txt',
            'dslr_reorgnized.txt',
            'webcam_reorgnized.txt'
        ],
        prefix=args.data.dataset.root_path)
    elif args.data.dataset.name == 'officehome':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['Art', 'Clipart', 'Product', 'Real_World'],
        files=[
            'Art.txt',
            'Clipart.txt',
            'Product.txt',
            'Real_World.txt'
        ],
        prefix=args.data.dataset.root_path)
    elif args.data.dataset.name == 'visda2017':
        dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['train', 'validation'],
        files=[
            'train/image_list.txt',
            'validation/image_list.txt',
        ],
        prefix=args.data.dataset.root_path)
        dataset.prefixes = [join(dataset.path, 'train'), join(dataset.path, 'validation')]
    else:
        raise Exception(f'dataset {args.data.dataset.name} not supported!')

    source_domain_name = dataset.domains[args.data.dataset.source]
    target_domain_name = dataset.domains[args.data.dataset.target]
    source_file = dataset.files[args.data.dataset.source]
    target_file = dataset.files[args.data.dataset.target]

    return dataset, source_domain_name, target_domain_name, source_file, target_file



def get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes):
    
    dataset, source_domain_name, target_domain_name, source_file, target_file = get_dataset_file(args)

    train_transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor()
    ])

    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])

    source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform, filter=(lambda x: x in source_classes))
    source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=test_transform, filter=(lambda x: x in source_classes))
    target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=train_transform, filter=(lambda x: x in target_classes))
    target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=test_transform, filter=(lambda x: x in target_classes))

    classes = source_train_ds.labels
    freq = Counter(classes)
    class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

    source_weights = [class_weight[x] for x in source_train_ds.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

    source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                                sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
    source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)
    target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                                num_workers=args.data.dataloader.data_workers, drop_last=True)
    target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)

    return source_train_dl, source_test_dl, target_train_dl, target_test_dl




def get_auroc_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes):
    
    dataset, source_domain_name, target_domain_name, source_file, target_file = get_dataset_file(args)

    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])

    source_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=test_transform, filter=(lambda x: x in source_classes))
    target_known_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=test_transform, filter=(lambda x: x in common_classes))
    target_unknown_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=test_transform, filter=(lambda x: x in target_private_classes))

    classes = source_ds.labels
    freq = Counter(classes)
    class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

    source_weights = [class_weight[x] for x in source_ds.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_ds.labels))

    source_dl = DataLoader(dataset=source_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)
    target_known_dl = DataLoader(dataset=target_known_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)
    target_unknown_dl = DataLoader(dataset=target_unknown_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)

    return source_dl, target_known_dl, target_unknown_dl