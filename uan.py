from data import *
from net import *
from lib import *
import time
import datetime
import logging

import torch
from torch import nn
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb

from models.uan import UAN
from utils.logging import logger_init
from utils.utils import seed_everything
from data import get_class_per_split, get_dataloaders

cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

logger = logging.getLogger(__name__)

gpu_ids = select_GPUs(args.misc.gpus)
output_device = gpu_ids[0]

## LOGGINGS ##
now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f'{args.log.root_dir}/{now}'
# init logger
logger_init(logger, log_dir)
# init tensorboard summarywriter
writer = SummaryWriter(log_dir)
# dump configs
with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

source_classes, target_classes, common_classes, source_private_classes, target_private_classes = get_class_per_split(args)
source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(source_classes, target_classes, common_classes, source_private_classes, target_private_classes)

# init model
logger.info('Init model...')
start_time = time.time()
model = UAN(args, source_classes)
end_time = time.time()
loading_time = end_time - start_time
logger.info(f'Done loading model. Total time {loading_time}')

feature_extractor = nn.DataParallel(model.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(model.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator = nn.DataParallel(model.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator_separate = nn.DataParallel(model.discriminator_separate, device_ids=gpu_ids, output_device=output_device).train(True)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    discriminator.load_state_dict(data['discriminator'])
    discriminator_separate.load_state_dict(data['discriminator_separate'])

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax',
                         'target_share_weight']) as target_accumulator, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            feature = feature_extractor.forward(im)
            feature, __, before_softmax, predict_prob = classifier.forward(feature)
            domain_prob = discriminator_separate.forward(__)

            target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                          class_temperature=1.0)

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    def outlier(each_target_share_weight):
        return each_target_share_weight < args.test.w_0

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

    for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label, target_share_weight):
        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if not outlier(each_target_share_weight[0]) and each_pred_id == each_label:
                counters[each_label].Ncorrect += 1.0
        else:
            counters[-1].Ntotal += 1.0
            if outlier(each_target_share_weight[0]):
                counters[-1].Ncorrect += 1.0

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    print(f'test accuracy is {acc_test.item()}')
    exit(0)

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator_separate = OptimWithSheduler(
    optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

current_epoch = 0
global_step = 0
best_acc = 0

# total steps / epochs
steps_per_epoch = min(len(source_train_dl), len(target_train_dl))
total_epoch = round(args.train.min_step / steps_per_epoch)
logger.info(f'Total epoch {total_epoch}, steps per epoch {steps_per_epoch}, total step {args.train.min_step}')

# log every epoch
log_interval = steps_per_epoch
# test every epoh
test_interval = steps_per_epoch

# total_steps = tqdm(range(args.train.min_step),desc='global step')

while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {current_epoch} ', total=steps_per_epoch)
    current_epoch += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        # fc1_s : (batch_size, 2048)
        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        # fc1_s                 : (batch, hidden_dim)
        # feature_source        : (batch, bottleneck_dim)
        # fc2_s                 : (batch, num_source_label)
        # predict_prob_source   : (batch, num_source_label)
        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        # shape : (batch, 1)
        domain_prob_discriminator_source = discriminator.forward(feature_source)
        # shape : (batch, 1)
        domain_prob_discriminator_target = discriminator.forward(feature_target)

        # shape : (batch, 1)
        domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
        # shape : (batch, 1)
        domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

        # shape : (batch, 1)
        source_share_weight = model.get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
        source_share_weight = model.normalize_weight(source_share_weight)
        # shape : (batch, 1)
        target_share_weight = model.get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
        target_share_weight = model.normalize_weight(target_share_weight)
            
        # ==============================compute loss
        adv_loss = torch.zeros(1, 1).to(output_device)
        adv_loss_separate = torch.zeros(1, 1).to(output_device)

        tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))

        # ============================== cross entropy loss
        ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
        ce = torch.mean(ce, dim=0, keepdim=True)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
            loss = ce + adv_loss + adv_loss_separate
            loss.backward()

        global_step += 1

        if global_step % log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            writer.add_scalar('adv_loss', adv_loss, global_step)
            writer.add_scalar('ce', ce, global_step)
            writer.add_scalar('adv_loss_separate', adv_loss_separate, global_step)
            writer.add_scalar('acc_train', acc_train, global_step)

            pdb.set_trace()

        if global_step % test_interval == 0:

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier, discriminator_separate], train=False) as mgr, \
                 Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax', 'target_share_weight']) as target_accumulator, \
                 torch.no_grad():

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.to(output_device)
                    label = label.to(output_device)

                    feature = feature_extractor.forward(im)
                    feature, __, before_softmax, predict_prob = classifier.forward(feature)
                    domain_prob = discriminator_separate.forward(__)

                    target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                                  class_temperature=1.0)

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            def outlier(each_target_share_weight):
                return each_target_share_weight < args.test.w_0

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

            for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label,
                                                                                 target_share_weight):
                if each_label in source_classes:
                    counters[each_label].Ntotal += 1.0
                    each_pred_id = np.argmax(each_predict_prob)
                    if not outlier(each_target_share_weight[0]) and each_pred_id == each_label:
                        counters[each_label].Ncorrect += 1.0
                else:
                    counters[-1].Ntotal += 1.0
                    if outlier(each_target_share_weight[0]):
                        counters[-1].Ncorrect += 1.0

            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)

            logger.add_scalar('acc_test', acc_test, global_step)
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
                'discriminator_separate': discriminator_separate.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)