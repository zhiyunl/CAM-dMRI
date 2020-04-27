"""
Class Activation Mapping For MRI image
"""
import os

from torch.backends import cudnn

from CAMGen import *
from HCPLoader import *
from models.alexnet import *
from models.googlenet import *
from models.inception import *
from models.vgg import *
from train import *
from utils import *

# transformation
normalize = transforms.Normalize(
    mean=[0.485],
    std=[0.225]
)

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize
])

if TRAIN:
    print("Training ")
    # Loading Training
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
        transforms.RandomApply([AddGaussianNoise(0., 0.2), transforms.RandomErasing()], p=0.3),
    ])
    transform_valid = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # transforms.RandomHorizontalFlip(p=0.1),
        # transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize
    ])
    train_loader, valid_loader = train_valid_loader(source, transform_train, transform_valid)
    print("Using Data Source: {}".format(source))
    print("Total Train Dataset: {}, with {}({}) as validation".format(len(train_loader.dataset),
                                                                      len(valid_loader) * BATCH_SIZE, RATIO))
# load test data
test_loader = test_loader(source, transform_test)
print("Total Test Dataset: {}".format(len(test_loader.dataset)))


def runOnce():
    print("Initializing Model : ", ModelN)
    if ModelN == "inception":
        net = inception_v3(num_classes=len(classes))
        final_conv = 'conv_final'
    elif ModelN == "googlenet":
        net = googlenet(num_classes=len(classes))
        final_conv = 'conv_final'
    elif ModelN == "alexnet":
        net = alexnet(num_classes=len(classes))
        final_conv = 'features'
    elif ModelN == "vgg":
        net = vgg_gap(num_classes=len(classes))
        final_conv = 'features'
    elif ModelN == 'googlenet-gmp':
        net = googlenet(num_classes=len(classes), gmp=True)
        final_conv = 'conv_final'
    else:
        net = alexnet(bn=True, num_class=2)
        final_conv = 'conv'

    if not TRAIN:
        for param in net.parameters():
            param.requires_grad = False

    if USE_CUDA:
        net.cuda()
        cudnn.benchmark = True

    # load checkpoint
    if RESUME:
        print("===> Resuming from checkpoint : checkpoint/" + ModelN + str(RESUME) + '.pt')
        assert os.path.isfile('checkpoint/' + ModelN + str(RESUME) + '.pt'), 'Error: no checkpoint found!'
        net.load_state_dict(torch.load('checkpoint/' + ModelN + str(RESUME) + '.pt'))

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    if TRAIN:
        print("start training")
        train(net, train_loader, valid_loader)
    else:
        print("Using pretrained network, testing only")
        test(test_loader, net)
        # CAM
        if CAM:
            print("hook feature extractor")
            # hook the feature extractor
            features_blobs = []

            net._modules.get(final_conv).register_forward_hook(hook_feature)
            cam_dir = "camIn"
            for root, subdirs, files in os.walk(cam_dir):
                for file in files:
                    get_cam(net, features_blobs, os.path.join(root, file))
                    # get_cam(net, features_blobs, os.path.join(root, file))


if __name__ == "__main__":
    # train all models
    for i in range(len(MODELS)):
        ModelN = MODELS[i]
        setPara("ModelN", ModelN)
        LOGDIR = "runs/{}".format(MODELS[i])
        setPara("LOGDIR", LOGDIR)
        writer = SummaryWriter(LOGDIR, comment=str(RESUME) + "_" + str(EPOCH))  # purge_step=RESUME + 1,
        setPara("writer", writer)
        runOnce()
        writer.close()
    # ModelN = "vgg"
    # setPara("ModelN", ModelN)
    # LOGDIR = "runs/{}".format(ModelN)
    # setPara("LOGDIR", LOGDIR)
    # writer = SummaryWriter(LOGDIR,  comment=str(RESUME) + "_" + str(EPOCH)) # purge_step=RESUME + 1,
    # setPara("writer", writer)
    # runOnce()
    # writer.close()
