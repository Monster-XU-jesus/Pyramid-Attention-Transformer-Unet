import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # 添加验证集
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                           transform=transforms.Compose(
                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of validation set is: {}".format(len(db_val)))
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=base_lr * 0.01)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    best_epoch = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 在每个epoch结束后评估模型性能
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            dice_score = 0
            for val_batch in valloader:
                val_images, val_labels = val_batch['image'], val_batch['label']
                val_images, val_labels = val_images.cuda(), val_labels.cuda()
                val_outputs = model(val_images)
                val_loss_ce = ce_loss(val_outputs, val_labels[:].long())
                val_loss_dice = dice_loss(val_outputs, val_labels, softmax=True)
                val_loss = 0.5 * val_loss_ce + 0.5 * val_loss_dice
                total_val_loss += val_loss.item()
                
                # 计算Dice分数作为性能指标
                val_outputs = torch.softmax(val_outputs, dim=1)
                val_outputs = torch.argmax(val_outputs, dim=1)
                dice_score += calculate_dice_score(val_outputs, val_labels, num_classes)
            
            avg_val_loss = total_val_loss / len(valloader)
            avg_dice_score = dice_score / len(valloader)
            
            writer.add_scalar('val/loss', avg_val_loss, epoch_num)
            writer.add_scalar('val/dice_score', avg_dice_score, epoch_num)
            logging.info('Epoch %d : val_loss : %f, val_dice_score: %f' % (epoch_num, avg_val_loss, avg_dice_score))
            
            # 保存最佳模型
            current_performance = avg_dice_score  # 使用Dice分数作为性能指标
            if current_performance > best_performance:
                best_performance = current_performance
                best_epoch = epoch_num
                best_model_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info("保存最佳模型到 {}，性能指标: {}".format(best_model_path, best_performance))

        # 在每个epoch结束时更新学习率
        # scheduler.step(loss)  # 使用当前epoch的损失来更新学习率

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            logging.info("最佳模型在第 {} 轮，性能指标: {}".format(best_epoch, best_performance))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

# 添加计算Dice分数的函数
def calculate_dice_score(pred, target, num_classes):
    """
    计算Dice分数
    :param pred: 预测结果 [B, H, W]
    :param target: 真实标签 [B, H, W]
    :param num_classes: 类别数量
    :return: 平均Dice分数
    """
    dice_score = 0
    for cls in range(1, num_classes):  # 跳过背景类
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union > 0:
            dice_score += (2.0 * intersection) / union
    
    return dice_score / (num_classes - 1)  # 平均Dice分数，不包括背景