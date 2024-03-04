import numpy as np, argparse, torch, os
import torch.nn as nn
from models import GenStudent, GenClassifier, vit_parameters, classifier_layels_list
from models.model import SpliceModel
from utils import int_list, float_list, save_model, save_data
from utils.dataset import get_datasets, image_transform
from utils.train import common_train, common_test
from models import VisionTransformer
from utils.draw_curves import draw_curves

def main(args):

    # student_model = GenStudent(vit_parameters, args.device)  # size of student model is 17,367,552
    classifier = GenClassifier(512, 10, classifier_layels_list, args.device)
    classifier.train()
    model = SpliceModel(classifier, vit_parameters).to(args.device)
    # model = VisionTransformer(input_resolution=224,patch_size=32,width=384,layers=9,heads=384//16,output_dim=512).to(args.device)

    if args.multi_gpu:
        model = nn.DataParallel(model)
    print(">> Training on: ", torch.cuda.current_device())
    # print(model)

    train_loader, test_loader = get_datasets(image_transform, args.batch_size, 'stl10')

    model, loss, acc, test_acc = common_train(model, train_loader, test_loader, args)
    common_test(model, test_loader, args)
    save_data(loss, './results/curve_data/cm_train_loss.txt')
    save_data(test_acc, './results/curve_data/cm_test_loss.txt')
    save_model(model, './results/weights/student_whole.pt')
    draw_curves(loss, './results/curves/cm_train_loss.png', True)
    draw_curves(acc, './results/curves/cm_train_acc.png', True)
    draw_curves(test_acc, './results/curves/cm_test_acc.png', True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch KD-Clip')
    ## Common arguments
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N', help='input batch size for training (default: [2048, 1024])')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: [10, 10])')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: [1e-3, 1e-3])')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training (default: cuda)')
    parser.add_argument('--multi-gpu', type=int, default=0, help='use multiple gpus for training')
    
    args = parser.parse_args()

    main(args)
