# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler, PartitianPrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from prototypical_loss import get_prob
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from parser import get_parser
import time

from tqdm import tqdm
import numpy as np
import torch
import os


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_list_dataloader(opt, mode, k):
    """
    [init_list_dataloader] will return a list of k-partitioned dataset.

    Args:
        opt: parameters
        mode: mode for dataloader
        k: the number of partitions

    Return:
        list of torch.utils.data.DataLoader for each random subset of data
    """
    out = []
    dataset = init_dataset(opt, mode)
    len_dataset = len(dataset.y)
    len_partition = np.ceil(len_dataset / k)
    indices = list(range(len_dataset))

    for idx in range(k):
        labels_k = dataset.y[idx*int(len_partition):(idx+1)*int(len_partition)+1]
        sampler_k = init_sampler(opt, labels_k, mode)
        dataloader_k = torch.utils.data.DataLoader(dataset, batch_sampler=sampler_k)
        out.append(dataloader_k)
    return out


def init_partition_sampler(opt, labels, mode, k, idx):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PartitianPrototypicalBatchSampler(labels=labels,
                                            classes_per_it=classes_per_it,
                                            num_samples=num_samples,
                                            iterations=opt.iterations,
                                            num_partitions=k,
                                            idx=idx)


def partitioned_dataloader(opt, mode, k):
    """
    Will return a list of dataloaders each has a partition of original dataset.
    The partitioning was done per class, ensuring that each dataloader will contain the full set of classes.

    args
        opt: parameters
        mode: just mode
        k: the number of partitions
    """
    dataloader_list = []
    dataset = init_dataset(opt, mode)
    for idx in range(1, k+1):
        sampler_idx = init_partition_sampler(opt, dataset.y, mode, k, idx)
        dataloader_idx = torch.utils.data.DataLoader(dataset, batch_sampler=sampler_idx)
        dataloader_list.append(dataloader_idx)

    return dataloader_list


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, model_name=None, teachers=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    if model_name is None:
        model_name = 'model'

    best_model_path = os.path.join(opt.experiment_root, 'best_'+model_name+'.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_'+model_name+'.pth')
    j = 0

    total_tr_time = 0
    total_val_time = 0

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        # j += 1
        # if j == 2:
        #     break

        start_epoch = time.time()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            if teachers is None:
                teacher_targets=None
            else:
                probs = []
                for teacher in teachers:

                    model_output_t = teacher(x)
                    probs.append(get_prob(model_output_t, y, opt.num_support_tr).unsqueeze(2)) # batch x class x 1
                probs = torch.cat(probs, dim=2) # batch x class x teacher
                teacher_targets = torch.mean(probs, dim=2) # batch x class

            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr,
                                teacher_targets=teacher_targets)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        tr_time = time.time()-start_epoch
        total_tr_time += tr_time
        print('Train-WallClock: {}'.format(tr_time))
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        start_val = time.time()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        val_time = time.time()-start_val
        total_val_time += val_time
        print('Val-WallClock: {}'.format(total_val_time))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    print('Avg-Train-WallClock: {}'.format(total_tr_time/opt.epoch))
    print('Avg-Val-WallClock: {}'.format(total_val_time/opt.epoch))

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    test_total = 0
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        start_test = time.time()
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_tr)
            avg_acc.append(acc.item())

        test_time = time.time()-start_test
        test_total += test_time
        print('Inference-WallClock: {}'.format(test_time))
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))
    print('Avg-Test-WallClock: {}'.format(test_total/10))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    # first train many weak techers
    tr_dataloaders = partitioned_dataloader(options, 'train', 3)
    val_dataloader = init_dataloader(options, 'val')
    # num_teacher = len(tr_dataloaders)

    best_teachers = []

    print('Start training teachers')
    for i, itr in enumerate(tr_dataloaders):
        model = init_protonet(options)
        optim = init_optim(options, model)
        lr_scheduler = init_lr_scheduler(options, optim)
        res = train(opt=options,
                    tr_dataloader=itr,
                    val_dataloader=val_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler,
                    model_name='teacher'+str(i)
                    )
        best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
        model.load_state_dict(best_state)
        best_teachers.append(model) # append best teacher
    print('Finished training teachers')


    print('Start training a student')
    tr_dataloader = init_dataloader(options, 'train')
    test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                teachers=best_teachers)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


if __name__ == '__main__':
    main()
