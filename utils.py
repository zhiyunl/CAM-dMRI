import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

classes = {0: 'young', 1: 'old'}

# TODO
CAM = True
USE_CUDA = torch.cuda.is_available()
RESUME = 40
EPOCH = 700
TRAIN = True
MODELS = ['alexnet', 'alexnet-bn', 'googlenet', 'googlenet-gmp', 'vgg', 'inception']  #
ModelN = "new"
source = "../DATA/HCP/16x3_slice/"
LOGDIR = "runs/{}".format(ModelN)
# writer = SummaryWriter(LOGDIR, purge_step=RESUME + 1, comment=str(RESUME) + "_" + str(EPOCH))
writer = None
criterion = torch.nn.CrossEntropyLoss()

# hyperparameters
BATCH_SIZE = 16
IMG_SIZE = 224
LEARNING_RATE = 0.001
RATIO = 0.2


def setPara(name, val):
    global LOGDIR
    global ModelN
    global writer
    if name == "ModelN":
        ModelN = val
    elif name == "LOGDIR":
        LOGDIR = val
    elif name == "writer":
        writer = val
    else:
        pass


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10, 10))
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx + 1, xticks=[], yticks=[])
        plt.imshow(images[idx][0].cpu(), cmap="gray")
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx].item()]),
            fontdict={'fontsize': 8, 'fontweight': 'medium'},
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    # writer.close()


def plot_curve():
    # plot
    train_loss = np.loadtxt("result/train_loss.txt", delimiter=",")
    train_acc = np.loadtxt("result/train_acc.txt", delimiter=",")
    valid_loss = np.loadtxt("result/valid_loss.txt", delimiter=",")
    valid_acc = np.loadtxt("result/valid_acc.txt", delimiter=",")

    plt.plot(train_acc, 'b', label="train")
    plt.plot(valid_acc, 'y', label="valid")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig('Acc-curve')
    plt.show()

    plt.plot(train_loss, 'b', label="train")
    plt.plot(valid_loss, 'y', label="valid")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig('Loss-curve')
    plt.show()


def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    # plt.show()


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
