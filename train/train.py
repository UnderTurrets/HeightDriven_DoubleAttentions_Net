import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from d2l import torch as d2l


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, scheduler, save_path, data_path,
               devices=d2l.try_all_gpus()):
    if os.path.exists(save_path) == True:
        pass
    else:
        os.makedirs(name=save_path)

    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.long(), loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                    epoch, i * len(features), len(train_iter.dataset),
                           100. * i / len(train_iter)))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()
        print(
            f"epoch {epoch + 1} --- loss {metric[0] / metric[2]:.3f} ---  train acc {metric[1] / metric[3]:.3f} --- test acc {test_acc:.3f} --- cost time {timer.sum()}")

        # ---------保存训练数据---------------
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch + 1)
        time_list.append(timer.sum())

        df = pd.DataFrame()
        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df['time'] = time_list
        df.to_excel(data_path)

        # ----------------保存模型-------------------
        if np.mod(epoch + 1, 10) == 0:
            torch.save(model.state_dict(), os.path.join(save_path,f'HDAnet_{epoch + 1}.pth'))
            import time

            # 防止显卡温度过高突然关机
            time.sleep(300)


if __name__ == "__main__":
    from db.camvid import train_loader, val_loader
    import network.HDAnet as net
    from conf import model_dict

    # 选择HAM层数
    i = 4
    model = net.HDAnet(num_classes=32, HAM_num=i).cuda()
    excel_path = "../res/HDAnet_" + str(i) + "HAM.xlsx"
    save_path = model_dict["HDANet_" + str(i) + "HAM"]["save_path"]

    model_file = os.path.join(model_dict["HDANet_" + str(i) + "HAM"]["save_path"],
                              model_dict["HDANet_" + str(i) + "HAM"]["model_file"])
    if (os.path.exists(model_file)):
        model.load_state_dict(torch.load(model_file), strict=True)
        print(f"success to load {model_file}")
    else:
        print(f"fail to load {model_file}")

    # 损失函数选用多分类交叉熵损失函数
    lossf = nn.CrossEntropyLoss(ignore_index=255)
    # 选用adam优化器来训练
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    train_ch13(net=model, train_iter=train_loader, test_iter=val_loader, loss=lossf, trainer=optimizer, num_epochs=30,
               scheduler=scheduler, save_path=save_path, data_path=excel_path)
