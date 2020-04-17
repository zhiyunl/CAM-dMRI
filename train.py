import numpy as np
import torch
from torch.nn import functional as F

import utils


def train(net, train_loader, valid_loader):
    optimizer = torch.optim.SGD(net.parameters(), lr=utils.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    for epoch in range(utils.RESUME * 10 + 1, utils.EPOCH + 1):
        trainOnce(train_loader, net, epoch, optimizer)
        validation(valid_loader, net, epoch)
        # if epoch % 30 == 0:
        #     utils.writer.close()
        #     utils.setPara("writer", SummaryWriter(utils.LOGDIR,  comment=str(utils.RESUME) + "_" + str(utils.EPOCH)))


def trainOnce(trainloader, model, epoch, optimizer):
    model.train()
    acc_list, loss_list = [], []
    for batch_idx, (data, target) in enumerate(trainloader):
        if utils.USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = utils.criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate accuracy
        _, argmax = torch.max(output, 1)
        accuracy = (argmax.squeeze() == target.data).float().mean()
        acc_list.append(accuracy.item())

        loss_list.append(loss.item())

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [Batch {}/{} ]\t Batch Loss: {:.3f}\tBatch Accuracy: {:.3f}%'.format(
                epoch, batch_idx, len(trainloader), loss.item(), accuracy.item() * 100))
    acc_avg = np.average(acc_list) * 100
    loss_avg = np.average(loss_list)
    print('\nTrain Epoch: {}\t Epoch Loss: {:.3f}\t Epoch Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))

    ##############################3
    ## TensorBoard Log
    ##################################
    info = {'loss/train': loss_avg, 'accuracy/train': acc_avg}
    for tag, value in info.items():
        utils.writer.add_scalar(tag, scalar_value=value, global_step=epoch)

    with open('result/train_acc.txt', 'a') as f:
        np.savetxt(f, [acc_avg], delimiter=",")
    f.close()
    with open('result/train_loss.txt', 'a') as f:
        np.savetxt(f, [loss_avg], delimiter=",")
    f.close()

    # history_acc = np.loadtxt("result/train_acc.txt", delimiter=",")
    # np.savetxt("result/train_acc.txt", np.append(history_acc,acc_list), delimiter=",")
    #
    # history_loss = np.loadtxt("result/train_loss.txt", delimiter=",")
    # np.savetxt("result/train_loss.txt", np.append(history_loss,loss_list), delimiter=",")


def validation(validloader, model, epoch):
    model.eval()
    # acc_list, loss_list = [], []
    cnt = 0
    correct = 0
    loss = 0
    for idx, (data, target) in enumerate(validloader):
        if utils.USE_CUDA:
            data, target = data.cuda(), target.cuda()
        # data, target = data.T, torch.Tensor(target)
        output = model(data)
        # sum up batch loss
        loss += utils.criterion(output, target).item()
        # get the index of the max log-probability
        _, argmax = torch.max(output, 1)
        correct += (argmax.squeeze() == target.data).float().cpu().sum()

        # pred = output.data.max(1, keepdim=True)[1]
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        cnt += len(target)
        # batch_acc = 100. * correct / testloader.batch_size
        # acc_list.append(batch_acc.item())

    acc_avg = 100. * correct / cnt
    loss_avg = loss / len(validloader)
    result = '\nValidation Epoch: Epoch loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss_avg, int(correct), cnt, acc_avg)
    print(result)

    ##############################3
    ## TensorBoard Log
    ##################################
    info = {'loss/validation': loss_avg, 'accuracy/validation': acc_avg}
    for tag, value in info.items():
        utils.writer.add_scalar(tag, scalar_value=value, global_step=epoch)
    # writer.add_figure('')
    # Save checkpoint.
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'checkpoint/' + utils.ModelN + str(epoch // 10) + '.pt')
        # it = iter(testloader)
        # data, target = it.next()
        # if USE_CUDA:
        #     data = data.cuda()
        # writer.add_figure('Valid/predictions vs. actuals',
        #                   plot_classes_preds(model, data, target),
        #                   global_step=epoch)

        with open('result/result.txt', 'a') as f:
            f.write(result)
        f.close()

    with open('result/valid_acc.txt', 'ab') as f:
        np.savetxt(f, [acc_avg], delimiter=",")
    f.close()
    with open('result/valid_loss.txt', 'ab') as f:
        np.savetxt(f, [loss_avg], delimiter=",")
    f.close()


def test(testloader, model):
    model.eval()
    test_loss = 0
    correct = 0
    for idx, (data, target) in enumerate(testloader):
        if utils.USE_CUDA:
            data, target = data.cuda(), target.cuda()
        # data, target = data.T, torch.Tensor(target)
        output = model(data)
        # sum up batch loss
        test_loss += utils.criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for images, labels in testloader:
            if utils.USE_CUDA:
                images = images.cuda()
            output = model(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # plot all the pr curves
    for i in range(len(utils.classes)):
        utils.add_pr_curve_tensorboard(i, test_probs, test_preds)

    it = iter(testloader)
    # plot_data = None
    # plot_label = None
    # idx = np.arange(utils.BATCH_SIZE)
    # shuffle(idx)
    # print(idx)
    # for i in range(16):
    #     data, target = it.next()
    #     if plot_data is None:
    #         plot_data_size = list(data[0].size())
    #         plot_data_size.insert(0, 16)
    #         plot_data = torch.empty(plot_data_size, dtype=data.dtype, device=data.device)
    #         plot_label = torch.empty(16, dtype=target.dtype, device=target.device)
    #     plot_data[i] = data[idx[i]]
    #     plot_label[i] = target[idx[i]]
    # print(plot_data)
    # if utils.USE_CUDA:
    #     plot_data = plot_data.cuda()
    plot_data, plot_label = it.next()
    utils.writer.add_figure('Test/predictions vs. actuals',
                            utils.plot_classes_preds(model, plot_data.cuda(), plot_label),
                            global_step=utils.EPOCH)
    utils.writer.add_graph(model, plot_data.cuda())

    # grid = torchvision.utils.make_grid(plot_data)
    # writer.add_figure('Test/images',grid)

    # plot_classes_preds(model,data,target)
    test_loss /= len(testloader.dataset)
    test_acc = 100. * correct / len(testloader.dataset)

    result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(testloader.dataset), test_acc)
    print(result)

    with open('result/result.txt', 'a') as f:
        f.write(result)
    f.close()
