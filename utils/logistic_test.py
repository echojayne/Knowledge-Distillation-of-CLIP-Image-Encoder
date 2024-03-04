from . import get_features
import torch, datetime
from sklearn.linear_model import LogisticRegression
from models import GenStudent, GenTeacher, vit_parameters
from utils.dataset import get_datasets, image_transform

def logic_test(model, train_loader, test_loader, device):
    
    print(f">> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running logistic regression on the features extracted from the model")
    # Calculate the image features
    train_features, train_labels = get_features(model, train_loader, device)
    test_features, test_labels = get_features(model, test_loader, device)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    train_acc = classifier.score(train_features, train_labels)
    test_acc = classifier.score(test_features, test_labels)

    print(f">> Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # only MSE loss
    # Teacher Model Accuracy = 88.762
    # Student Model Accuracy = 81.308

    return train_acc, test_acc

if __name__ == '__main__':
    device = 'cuda:1'
    model = GenTeacher('ViT-B/32', device)  # size of teacher model is 87,849,216
    # 加载权重
    # model.load_state_dict(torch.load('./results/weights/student.pt'))
    train_loader, test_loader = get_datasets(image_transform, 32, 'stl10')
    logic_test(model, train_loader, test_loader, device)
