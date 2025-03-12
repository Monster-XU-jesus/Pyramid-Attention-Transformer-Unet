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
    max_iterations = args.max_epochs * len(trainloader)
    warmup_iterations = args.warmup_epochs * len(trainloader)  # 新增warmup迭代计算
    logging.info(f"Warmup iterations: {warmup_iterations}")
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = float('inf')  # 修改初始化值为正无穷
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        epoch_loss = 0.0  # 新增epoch_loss统计
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # 将损失函数中的Dice权重从0.5提升到0.7
            loss = 0.3 * loss_ce + 0.7 * loss_dice  # 增强Dice的影响 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # 累计epoch总loss
            # 修改学习率调度逻辑
            if iter_num < warmup_iterations:
                # Warmup阶段：线性增长
                lr_ = base_lr * (iter_num / warmup_iterations)
            else:
                # 原多项式衰减策略
                progress = (iter_num - warmup_iterations) / (max_iterations - warmup_iterations)
                lr_ = base_lr * (1.0 - progress) ** 0.9
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            with torch.no_grad():
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                targets = label_batch
                
                # 计算IoU（交并比）
                intersection = (preds & targets).float().sum((1, 2))
                union = (preds | targets).float().sum((1, 2))
                iou = (intersection + 1e-6) / (union + 1e-6)
                
                # 计算Dice系数（实际值非损失）
                dice_coeff = (2.0 * intersection) / (preds.float().sum((1, 2)) + targets.float().sum((1, 2)) + 1e-6)
                
                # 计算像素精度
                correct = (preds == targets).float().sum()
                total = torch.numel(preds)
                pixel_acc = correct / total

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('metrics/dice_loss', loss_dice, iter_num) # 了解有什么作用
            writer.add_scalar('metrics/mean_iou', iou.mean(), iter_num)
            writer.add_scalar('metrics/mean_dice', dice_coeff.mean(), iter_num) 
            writer.add_scalar('metrics/pixel_accuracy', pixel_acc, iter_num)
            writer.add_scalar('metrics/class_iou', iou[1], iter_num)  # 示例记录第1类iou

            logging.info('iteration %d : loss: %f, ce_loss: %f, dice_loss: %f' % 
                        (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 在每个epoch结束时更新学习率
        # scheduler.step(loss)  # 使用当前epoch的损失来更新学习率

        # ============ 新增模型保存逻辑 ============
        # 计算epoch平均loss
        epoch_avg_loss = epoch_loss / len(trainloader)
        
        # 保存最佳模型
        if epoch_avg_loss < best_performance:
            best_performance = epoch_avg_loss
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Save best model with loss {epoch_avg_loss:.4f}")
            
        # 保存最新模型
        save_mode_path = os.path.join(snapshot_path, 'latest_model.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("Save latest model")

        # 原有的模型保存逻辑保持不变
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"