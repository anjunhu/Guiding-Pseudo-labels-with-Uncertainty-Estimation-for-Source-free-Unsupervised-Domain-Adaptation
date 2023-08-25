import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from model import *
import argparse
import numpy as np
import torchvision.transforms as transforms
import wandb
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import *
from utils import *
from datasets import *
from evaluation import *
from os.path import join
from pprint import pprint

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--dataset', default='visdac/source', type=str)
parser.add_argument('--num_class', default=10, type=int)

parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--train_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
parser.add_argument('--test_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
parser.add_argument('--flag', default='', choices=['', 'Younger', 'Older'], type=str, help='')

parser.add_argument('--alfa', default=0.1, type=float)

parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

parser.add_argument('--run_name', type=str)
parser.add_argument('--wandb', action='store_true', help="Use wandb")

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.wandb:
    wandb.init(project="Guiding Pseudo-labels with Uncertainty Estimation for Test-Time Adaptation", name = args.run_name)


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1)
    return loss

# Training
def train(epoch, net, optimizer, trainloader):
    loss = []
    acc = []
    tol_outputs = []
    tol_targets = []
    tol_sensitive = []
    group_metrics = {}
    for sa in range(5):
        group_metrics[f'Training loss_cls_ce A{sa}'] = 0.0
        group_metrics[f'Training auc A{sa}'] = 0.0
        group_metrics[f'Training acc A{sa}'] = 0.0

    net.train()

    for batch_idx, batch in enumerate(trainloader): 
        x = batch[0].cuda()
        y = batch[2].cuda()
        sensitive = batch[6].cuda()
        if batch_idx==0:
            print('\nTraining Batch Y0/Y1', x[y.squeeze()==0].shape, x[y.squeeze()==1].shape)
            for sa in range(5):
                print(f'A{sa} Total/Y0/Y1:', x[sensitive.squeeze()==sa].shape,
                                             x[torch.logical_and((sensitive.squeeze()==sa),(y.squeeze()==0))].shape,
                                             x[torch.logical_and((sensitive.squeeze()==sa),(y.squeeze()==1))].shape)

        _, outputs = net(x)

        l = smoothed_cross_entropy(outputs, y, args.num_class, args.alfa).mean()

        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        accuracy = accuracy_score(y.to('cpu'), outputs.to('cpu').max(1)[1])

        loss.append(l.item()) 
        acc.append(accuracy)
        tol_targets.append(y.detach())
        tol_sensitive.append(sensitive.detach())
        tol_outputs.append(outputs.detach())
        if batch_idx % 100 == 0:
            print('Epoch [%3d/%3d] Iter[%3d/%3d]\t ' 
                %(epoch, args.num_epochs, batch_idx+1, len(trainloader)))

    loss = np.mean( np.array(loss) )
    acc = np.mean( np.array(acc) )
    tol_outputs = torch.cat(tol_outputs).detach().cpu()
    tol_targets = torch.cat(tol_targets).cpu()
    auc = calculate_auc(F.softmax(tol_outputs, dim=1)[:, 1], tol_targets)
    tol_outputs = tol_outputs
    tol_sensitive = torch.cat(tol_sensitive).cpu()
    for sa in range(5):
        if len(tol_sensitive.squeeze()==sa) > 0:
            logits_sa = tol_outputs[tol_sensitive.squeeze()==sa]
            y_sa = tol_targets[tol_sensitive.squeeze()==sa]
            loss_cls_ce_sa = smoothed_cross_entropy(logits_sa, y_sa, args.num_class, args.alfa).mean()
            group_metrics[f'Training loss_cls_ce A{sa}'] = float(loss_cls_ce_sa.detach().item())
            group_metrics[f'Training acc A{sa}'] = accuracy_score(y_sa.to('cpu'), logits_sa.to('cpu').max(1)[1])
            group_metrics[f'Training auc A{sa}'] = calculate_auc(F.softmax(logits_sa, dim=1)[:, 1].detach().cpu(), y_sa.to('cpu'))

    print("Training acc = ", acc)
    pprint(group_metrics)
    if args.wandb:
        wandb.log({
        'train_loss': loss, \
        'train_acc': acc, \
        }, step=epoch)
        wandb.log(group_metrics, step=epoch)

def test(epoch,net):
    net.eval()
    correct = 0

    total = 0
    it = 0
    tol_outputs = []
    tol_targets = []
    tol_sensitive = []
    tol_indices = []
    group_metrics = {}
    loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets, idxs, sensitive = batch[0].cuda(), batch[2].cuda(), batch[3].cuda(), batch[6].cuda()

            if batch_idx==0:
                print('\nValidation Batch Y0/Y1', inputs[targets.squeeze()==0].shape, inputs[targets.squeeze()==1].shape)
                for sa in range(5):
                    print(f'A{sa} Total/Y0/Y1:', inputs[sensitive.squeeze()==sa].shape,
                                             inputs[torch.logical_and((sensitive.squeeze()==sa),(targets.squeeze()==0))].shape,
                                             inputs[torch.logical_and((sensitive.squeeze()==sa),(targets.squeeze()==1))].shape)

            _, outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            loss += CEloss(outputs, targets)
            it += 1

            tol_targets.append(targets.detach())
            tol_sensitive.append(sensitive.detach())
            tol_indices.append(idxs.detach())
            tol_outputs.append(outputs.detach())

    loss = loss/it
    tol_outputs = torch.cat(tol_outputs).cpu()
    tol_sensitive = torch.cat(tol_sensitive).cpu()
    tol_targets = torch.cat(tol_targets).cpu()
    tol_indices = torch.cat(tol_indices).cpu()
    probs = F.softmax(tol_outputs, dim=1)[:, 1].detach().cpu()
    auc = calculate_auc(probs, tol_targets)
    acc = accuracy_score(tol_targets, tol_outputs.max(1)[1])
    for sa in range(5):
        if len(tol_sensitive.squeeze()==sa) > 0:
            logits_sa = tol_outputs[tol_sensitive.squeeze()==sa]
            y_sa = tol_targets[tol_sensitive.squeeze()==sa]
            loss_cls_ce_sa = smoothed_cross_entropy(logits_sa, y_sa, args.num_class, args.alfa).mean()
            group_metrics[f'Validation loss_cls_ce A{sa}'] = float(loss_cls_ce_sa.detach().item())
            group_metrics[f'Validation acc A{sa}'] = accuracy_score(y_sa.to('cpu'), logits_sa.to('cpu').max(1)[1])
            group_metrics[f'Validation auc A{sa}'] = calculate_auc(F.softmax(logits_sa, dim=1)[:, 1].detach().cpu(), y_sa.to('cpu'))

    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  

    log_dict, t_predictions, pred_df = calculate_metrics(probs.numpy(), tol_targets.float().numpy(), tol_sensitive.numpy(), tol_indices.numpy(), sens_classes=5)
    log_dict['val_accuracy'] = acc
    log_dict['val_auc'] = auc
    if args.wandb:
        wandb.log(log_dict, step=epoch)

    return acc

def create_model(arch, args):
    model = Resnet(arch, args)

    model = model.cuda()
    return model

arch = 'resnet18'

if args.dataset.split('/')[0] == 'pacs':
    train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'PACS'),
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'PACS'),
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset.split('/')[0] == 'visdac':
    train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'VISDA-C'),
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'VISDA-C'),
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

    arch = 'resnet101'

elif 'domainnet' in args.dataset.split('/')[0]:
    train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, args.dataset.split('/')[0]),
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, args.dataset.split('/')[0]),
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

    arch = 'resnet50'

logdir = 'logs/' + args.run_name
net = create_model(arch, args)

cudnn.benchmark = True
train_sampler = None
test_sampler = None
g = torch.Generator()
g.manual_seed(args.seed)
if args.train_resampling and args.train_resampling != 'natural':  
    weights = train_dataset.get_weights(args.train_resampling, flag=args.flag)
    train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
if args.test_resampling and args.test_resampling != 'natural':
    test_weights = test_dataset.get_weights(args.test_resampling, flag=args.flag)
    test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True, generator=g)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              sampler=test_sampler,
                                              num_workers=4,
                                              drop_last=True,
                                              shuffle=False)

optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.5, nesterov=False)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

best = 0

for epoch in range(args.num_epochs+1):
 
    print('Train Nets')
    train(epoch, net, optimizer, train_loader) # train net1  

    acc = test(epoch,net) 

    if acc > best:
        save_weights(net, epoch, logdir + '/weights_best.tar')
        best = acc
        print("Saving best!")

        if args.wandb:
            wandb.run.summary['best_acc'] = best

