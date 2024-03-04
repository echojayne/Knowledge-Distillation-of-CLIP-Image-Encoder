import numpy as np, argparse, torch, os
import torch.nn as nn
from models import GenTeacher, GenStudent, GenClassifier, vit_parameters, classifier_layels_list
from utils import int_list, float_list, save_model, save_data
from utils.dataset import get_datasets, image_transform
from utils.train import pre_train, student_kd, test
from utils.logistic_test import logic_test
from utils.draw_curves import draw_curves

def main(args):

    teacher_model = GenTeacher(args.teacher_model, args.device)      # size of teacher model is 87,849,216
    student_model = GenStudent(vit_parameters, args.device)  # size of student model is 17,367,552
    student_model.load_state_dict(torch.load('/home/ubuntu/users/dky/CLIP-KD/results/weights/student_stl10.pt'), strict=False)
    classifier_tec = GenClassifier(512, 10, classifier_layels_list, args.device)
    classifier_stu = GenClassifier(512, 10, classifier_layels_list, args.device)

    if args.multi_gpu:
        teacher_model = nn.DataParallel(teacher_model)
        student_model = nn.DataParallel(student_model)
        classifier_tec = nn.DataParallel(classifier_tec)
        classifier_stu = nn.DataParallel(classifier_stu)
    print(">> Training on: ", end=' ')
    for i in range(torch.cuda.device_count()):    
        print(torch.cuda.get_device_name(i), end=',')

    train_loader, test_loader = get_datasets(image_transform, args.batch_size[-1], 'stl10')

    # Classifier for pre-training teacher models
    # classifier_tec, ACC, LOSS = pre_train(teacher_model, classifier_tec, train_loader, args)
    # test(teacher_model, classifier_tec, test_loader, args)
    # save_data(ACC, '/home/ubuntu/users/dky/CLIP-KD/results/curve_data/pre_train_acc.txt')
    # save_data(LOSS, '/home/ubuntu/users/dky/CLIP-KD/results/curve_data/pre_train_loss.txt')
    # save_model(classifier, '/home/ubuntu/users/dky/CLIP-KD/results/weights/classifier.pt')

    classifier_tec.load_state_dict(torch.load('/home/ubuntu/users/dky/CLIP-KD/results/weights/classifier_tea_stl10.pt'), strict=False)
    test(teacher_model, classifier_tec, test_loader, args)
    student_model, LOSS_kd = student_kd(student_model, teacher_model, train_loader, args)
    test(student_model, classifier_tec, test_loader, args)
    classifier_stu, ACC, LOSS = pre_train(student_model, classifier_stu, train_loader, args)
    test(student_model, classifier_stu, test_loader, args)
    save_data(LOSS_kd, '/home/ubuntu/users/dky/CLIP-KD/results/curve_data/kd_train_loss_stl10.txt')
    save_data(ACC, '/home/ubuntu/users/dky/CLIP-KD/results/curve_data/train_classifier_stu_acc_stl10.txt')
    save_data(LOSS, '/home/ubuntu/users/dky/CLIP-KD/results/curve_data/train_classifier_stu_loss_stl10.txt')
    save_model(student_model, '/home/ubuntu/users/dky/CLIP-KD/results/weights/student_stl10.pt')
    save_model(classifier_stu, '/home/ubuntu/users/dky/CLIP-KD/results/weights/classifier_stu_stl10.pt')
    draw_curves(LOSS_kd, '/home/ubuntu/users/dky/CLIP-KD/results/curves/kd_train_loss_stl10.png', True)
    draw_curves(ACC, '/home/ubuntu/users/dky/CLIP-KD/results/curves/train_classifier_stu_acc_stl10.png', False)
    draw_curves(LOSS, '/home/ubuntu/users/dky/CLIP-KD/results/curves/train_classifier_stu_loss_stl10.png', False)
    # logic_test(student_model, train_loader, test_loader, args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch KD-Clip')
    ## Common arguments
    parser.add_argument('--batch-size', type=int_list, default=[2048, 1024], metavar='N', help='input batch size for training (default: [2048, 1024])')
    parser.add_argument('--epochs', type=int_list, default=[10, 10], metavar='N', help='number of epochs to train (default: [10, 10])')
    parser.add_argument('--lr', type=float_list, default=[1e-3, 1e-3], metavar='LR', help='learning rate (default: [1e-3, 1e-3])')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training (default: cuda)')
    parser.add_argument('--multi-gpu', type=int, default=0, help='use multiple gpus for training')

    ## Arguments for KD-CLIP
    parser.add_argument('--teacher-model', type=str, default='ViT-B/32', help='teacher model name (default: ViT-B/32)')
    
    args = parser.parse_args()

    main(args)
