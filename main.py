import os
import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from models.slowfast_base import create_slowfast_base
from data.dataset import VideoDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred_labels = pred[:, 0]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    res.append(pred_labels)
    return res


def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Measure accuracy and record loss
        prec1, _, pred_labels = accuracy(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % 100 == 0:
            print(f'Epoch: [{epoch}][{step + 1}/{len(train_dataloader)}]')
            print(f'Loss {losses.avg:.4f}\t'
                  f'Acc@1 {top1.avg:.3f}')

    # Log to tensorboard
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_acc', top1.avg, epoch)

    return top1.avg, losses.avg


def validate(model, val_dataloader, epoch, criterion, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Measure accuracy and record loss
            prec1, _, pred_labels = accuracy(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        print(f'Validation Results - Epoch: [{epoch}]')
        print(f'Loss {losses.avg:.4f}\t'
              f'Acc@1 {top1.avg:.3f}')

    # Log to tensorboard
    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_acc', top1.avg, epoch)

    return top1.avg, losses.avg


def main():
    # Set random seed for reproducibility
    torch.manual_seed(3407)
    cudnn.benchmark = False

    # Setup tensorboard
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_dir = os.path.join('logs', cur_time)
    writer = SummaryWriter(log_dir=log_dir)

    # Create model save directory
    model_save_dir = os.path.join('checkpoints', cur_time)
    os.makedirs(model_save_dir, exist_ok=True)

    # Dataset loading code here
    # Replace with your actual dataset and dataloader creation
    dataset = VideoDataset(directory='data', clip_len=64, frame_sample_rate=1)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Create model
    model = create_slowfast_base(num_classes=6)  # Replace with your model
    model = model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,  # Learning rate
                          momentum=0.9,
                          weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=30,  # Decay steps
                                          gamma=0.1)  # Decay rate

    best_acc = 0
    for epoch in range(100):  # Number of epochs
        print(f"Epoch: {epoch}")

        # Train for one epoch
        train_acc, train_loss = train(model, train_dataloader, epoch,
                                      criterion, optimizer, writer)
        print("train_acc:",train_acc)
        print("train_loss:",train_loss)

        # Evaluate on validation set
        val_acc, val_loss = validate(model, val_dataloader, epoch,
                                     criterion, writer)

        # Save checkpoint if best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(model_save_dir, 'best_model.pth'))

        scheduler.step()

    writer.close()
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()