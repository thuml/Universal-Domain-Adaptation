
from os.path import join

from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.data import Dataset

# from : https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L107
# data iterator that will never stop producing data
class ForeverDataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        
        return data

    def __len__(self):
        return len(self.dataloader)

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
    elif args.data.dataset.name == 'office-home':
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
    elif args.data.dataset.name == 'visda':
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



def get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes, return_id=False):
    
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
                                transform=train_transform, return_id=return_id, filter=(lambda x: x in source_classes))
    source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=test_transform, return_id=False, filter=(lambda x: x in source_classes))
    target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=train_transform, return_id=return_id, filter=(lambda x: x in target_classes))
    target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=test_transform, return_id=False, filter=(lambda x: x in target_classes))
    
    print(f'\n\nsource train : {len(source_train_ds)}')
    print(f'target train : {len(target_train_ds)}')
    print(f'target test  : {len(target_test_ds)}\n\n')

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
    
    
    print(f'\n\nsource train steps : {len(source_train_dl)}')
    print(f'target train steps : {len(target_train_dl)}')
    print(f'target test steps  : {len(target_test_dl)}\n\n')


    return source_train_dl, source_test_dl, target_train_dl, target_test_dl



# get special dataloader for calculating auroc
def get_auroc_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes):
    
    dataset, source_domain_name, target_domain_name, source_file, target_file = get_dataset_file(args)

    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])

    # TODO : source_classes vs source_private_classes
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




########################
#                      #
#     ONLY FOR CMU     #
#                      #
########################

from torchvision.transforms.transforms import *


def get_transforms():
    train_transform1 = Compose([
        Resize(256),
        RandomHorizontalFlip(),
        RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                    fillcolor=(255, 255, 255)),
        CenterCrop(224),
        RandomGrayscale(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    train_transform2 = Compose([
        Resize(256),
        RandomHorizontalFlip(),
        RandomPerspective(),
        FiveCrop(224),
        Lambda(lambda crops: crops[0]),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    train_transform3 = Compose([
        Resize(256),
        RandomHorizontalFlip(),
        RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                    fillcolor=(255, 255, 255)),
        FiveCrop(224),
        Lambda(lambda crops: crops[1]),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    train_transform4 = Compose([
        Resize(256),
        RandomHorizontalFlip(),
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, resample=Image.BICUBIC,
                    fillcolor=(255, 255, 255)),
        RandomPerspective(),
        FiveCrop(224),
        Lambda(lambda crops: crops[2]),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    train_transform5 = Compose([
        Resize(256),
        RandomHorizontalFlip(),
        RandomPerspective(),
        FiveCrop(224),
        Lambda(lambda crops: crops[3]),
        RandomGrayscale(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    return train_transform1, train_transform2, train_transform3, train_transform4, train_transform5


# get augmented source images for ensemble
def esem_dataloader(args, source_classes):
    
    dataset, source_domain_name, target_domain_name, source_file, target_file = get_dataset_file(args)

    train_transform1, train_transform2, train_transform3, train_transform4, train_transform5 = get_transforms()

    ds1 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform1, filter=(lambda x: x in source_classes))
    ds2 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform2, filter=(lambda x: x in source_classes))
    ds3 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform3, filter=(lambda x: x in source_classes))
    ds4 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform4, filter=(lambda x: x in source_classes))
    ds5 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform5, filter=(lambda x: x in source_classes))
    
    dl1 = DataLoader(dataset=ds1, batch_size=args.data.dataloader.batch_size,
                            num_workers=args.data.dataloader.data_workers, drop_last=True)
    dl2 = DataLoader(dataset=ds2, batch_size=args.data.dataloader.batch_size,
                            num_workers=args.data.dataloader.data_workers, drop_last=True)
    dl3 = DataLoader(dataset=ds3, batch_size=args.data.dataloader.batch_size,
                            num_workers=args.data.dataloader.data_workers, drop_last=True)
    dl4 = DataLoader(dataset=ds4, batch_size=args.data.dataloader.batch_size,
                            num_workers=args.data.dataloader.data_workers, drop_last=True)
    dl5 = DataLoader(dataset=ds5, batch_size=args.data.dataloader.batch_size,
                            num_workers=args.data.dataloader.data_workers, drop_last=True)

    return dl1, dl2, dl3, dl4, dl5



########################
#                      #
#    ONLY FOR UniOT    #
#                      #
########################


def get_dataloaders_for_uniot(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes):
    
        
    # target-private label
    tp_classes = sorted(set(target_classes) - set(source_classes))
    # source-private label
    sp_classes = sorted(set(source_classes) - set(target_classes))
    # common label
    common_classes = sorted(set(source_classes) - set(sp_classes))

    classes_set = {
        'source_classes': source_classes,
        'target_classes': target_classes,
        'tp_classes': tp_classes,
        'sp_classes': sp_classes,
        'common_classes': common_classes
    }

    uniformed_index = len(classes_set['source_classes'])

    dataset, source_domain_name, target_domain_name, source_file, target_file = get_dataset_file(args)

    # train_transform = Compose([
    #     Resize(256),
    #     RandomCrop(224),
    #     RandomHorizontalFlip(),
    #     ToTensor(),
    #     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    # test_transform = Compose([
    #     Resize(256),
    #     CenterCrop(224),
    #     ToTensor(),
    #     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    train_transform = Compose([
        Resize((256, 256)),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        Resize((256, 256)),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform, return_id=True, filter=(lambda x: x in source_classes))
    source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=test_transform, return_id=False, filter=(lambda x: x in source_classes))
    target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=train_transform, return_id=True, filter=(lambda x: x in target_classes))
    target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=test_transform, return_id=False, filter=(lambda x: x in target_classes))
    
    print(f'\n\nsource train : {len(source_train_ds)}')
    print(f'target train : {len(target_train_ds)}')
    print(f'target test  : {len(target_test_ds)}\n\n')

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
        
    # for memory queue init
    target_initMQ_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size, shuffle=True,
                                num_workers=1, drop_last=True)
    
    
    print(f'\n\nsource train steps : {len(source_train_dl)}')
    print(f'target train steps : {len(target_train_dl)}')
    print(f'target test steps  : {len(target_test_dl)}\n\n')


    return source_train_dl, source_test_dl, target_train_dl, target_test_dl, target_initMQ_dl, classes_set