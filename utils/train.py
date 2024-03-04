import torch, datetime
from tqdm.auto import trange
from tqdm import tqdm
import torch.nn.functional as F

def pre_train(teacher_model, classifier, train_loader, args):
    """
    Pre-train the classifier with teacher model

    Args:
        teacher_model (nn.Module): teacher model
        classifier (nn.Module): classifier model
        train_loader (DataLoader): training data loader
        args (argparse): arguments

    Returns:
        classifier (nn.Module): pre-trained classifier model

    """
    theTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr[0])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()

    teacher_model.eval()
    classifier.train()

    ACC = []
    LOSS = []
    for epoch in trange(args.epochs[0], desc=f'{theTime} Pre-Train Epoch', ncols=120):
        train_loader = tqdm(train_loader, leave=False, ncols=120)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_features = teacher_model(images)
            pre = classifier(teacher_features)

            loss = criterion(pre, labels)
            acc = (pre.argmax(dim=1) == labels).float().mean()
            loss.backward()
            optimizer.step()
            ACC.append(acc.item())
            LOSS.append(loss.item())

            train_loader.set_description(f'Pre-Train Batch')
            train_loader.set_postfix(loss=loss.item(), acc=acc.item())
            # if i % 20 == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
        train_loader.close()
        # scheduler.step()

    return classifier, ACC, LOSS

def student_kd(student_model, teacher_model, train_loader, args):
    """
    Train the student model with KD-CLIP

    Args:
        student_model (nn.Module): student model
        classifier (nn.Module): classifier model
        train_loader (DataLoader): training data loader
        args (argparse): arguments

    Returns:
        student_model (nn.Module): trained student model

    """
    theTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr[-1])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs[-1], eta_min=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    criterion = F.mse_loss

    teacher_model.eval()
    student_model.train()

    LOSS = []

    for epoch in trange(args.epochs[-1], desc=f'{theTime} KD-Train Epoch', ncols=120):
        train_loader = tqdm(train_loader, leave=False, ncols=120)
        for i, (images, _) in enumerate(train_loader):
            images = images.to(args.device)
            # labels = labels.to(args.device)

            teacher_features = teacher_model(images)
            student_features = student_model(images)

            loss = criterion(student_features, teacher_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            LOSS.append(loss.item())
            train_loader.set_description(f'KD-Train Batch')
            train_loader.set_postfix(loss=loss.item())#,lr=scheduler.get_last_lr())

            # if i % 20 == 0:
            #     print(f"Epoch {epoch+1}/{args.epochs} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        train_loader.close()
      #  scheduler.step()
    
    return student_model, LOSS

def common_train(model, train_loader, test_loader, args):
    """
    Train the student model with KD-CLIP

    Args:
        model (nn.Module): model to train
        train_loader (DataLoader): training data loader
        test_loader (DataLoader): testing data loader
        args (argparse): arguments

    Returns:
        None

    """
    theTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss() 

    LOSS = []
    ACC = []
    Test_ACC = []

    for epoch in trange(args.epochs, desc=f'{theTime} CM-Train Epoch', ncols=120):
        train_loader = tqdm(train_loader, leave=False, ncols=120)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            pre = model(images)
            acc = (pre.argmax(dim=1) == labels).sum().item()/len(pre)
            loss = criterion(pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
            ACC.append(acc)
            train_loader.set_description(f'CM-Train Batch')
            train_loader.set_postfix(loss=loss.item(), acc=acc)
            if epoch % 10 == 0:
                test_acc = common_test(model, test_loader, args)
                Test_ACC.append(test_acc)
        train_loader.close()
    
    return model, LOSS, ACC, Test_ACC


def test(model, classifier, test_loader, args):
    """
    Test the student model with classifier

    Args:
        model (nn.Module): model to test
        classifier (nn.Module): classifier model
        test_loader (DataLoader): testing data loader
        args (argparse): arguments

    Returns:
        None

    """ 
    theTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    model.eval()
    classifier.eval()

    correct = 0
    total = 0
    test_loader = tqdm(test_loader, ncols=120)
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_loader):

            images = images.to(args.device)
            labels = labels.to(args.device)


            features = model(images)
            outputs = classifier(features)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            test_loader.set_description(f'{theTime} Test on testset [{i+1}/{len(test_loader)}]')
            test_loader.set_postfix(acc=correct.item()/total)
            
    print(f"Accuracy: {100 * correct / total :.3f}%\n")

def common_test(model, test_loader, args):
    """
    Test the student model with classifier

    Args:
        model (nn.Module): model to test
        test_loader (DataLoader): testing data loader
        args (argparse): arguments

    Returns:
        None

    """
    theTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    model.eval()

    correct = 0
    total = 0
    test_loader = tqdm(test_loader, ncols=120)
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_loader):

            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            test_loader.set_description(f'{theTime} Test on testset [{i+1}/{len(test_loader)}]')
            test_loader.set_postfix(acc=correct.item()/total)
    test_loader.close()
    return 100 * correct / total
    # print(f"Accuracy: {100 * correct / total :.3f}%\n")