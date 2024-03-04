from utils import get_features
import torch, datetime
from sklearn.linear_model import LogisticRegression
from models import GenTeacher, GenStudent, vit_parameters
from utils.dataset import get_datasets, image_transform
from utils.logistic_test import logic_test
from utils.draw_curves import draw_curves
import torch
from ptflops import get_model_complexity_info
from torchstat import stat
if __name__ == '__main__':
    
    # draw_curves('results/curve_data/cm_train_loss.txt', 'results/curves/cm_train_loss.png', False)
    # 加载模型
    model_student = GenStudent(vit_parameters, 'cuda:2')
    model_teacher = GenTeacher('ViT-B/32', 'cuda:2')
    # 加载权重
    model_student.load_state_dict(torch.load('/home/ubuntu/users/dky/CLIP-KD/results/weights/student_84.7.pt'), strict=False)

    stat(model_student, (3, 224, 224))

    # train_loader, test_loader = get_datasets(image_transform, 32, 'stl10')
    # logic_test(model_student, train_loader, test_loader, 'cuda:2')
    # logic_test(model_teacher, train_loader, test_loader, 'cuda:2')
