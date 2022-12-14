

import mlfoundry as mlf
client = mlf.get_client()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name", type=str, required=True, help="name of run"
)
args = parser.parse_args()


import json
import random
from types import SimpleNamespace
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler, Subset


run = client.create_run(project_name="fashion-mnist-fe", run_name=args.run_name)
run.set_tags({"framework": "pytorch", "model-type": "cnn"})



transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST('fashion-mnist-data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('fashion-mnist-data', train=False, transform=transform)



# Log dataset folder as artifact
#run.log_artifact("fashion-mnist-data/FashionMNIST/gz/", artifact_path="fashion-mnist-data")


# In[8]:


idx2label = [
  "T-shirt/Top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat", 
  "Sandal", 
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle Boot"
]



# Log labels as artifact
with open("config.json", "w") as f:
    json.dump({"idx2label": idx2label}, f)
run.log_artifact("config.json", artifact_path="")


y_train = pd.DataFrame([y for _, y in train_dataset], columns=["y"])
y_test = pd.DataFrame([y for _, y in test_dataset], columns=["y"])
train_sample = y_train.groupby('y').sample(n=3, random_state=42).index
test_sample = y_test.groupby('y').sample(n=3, random_state=42).index


# In[11]:


fig = plt.figure(figsize=(30, 10))
for i, idx in enumerate(train_sample):
    image_t, label_idx = train_dataset[idx]
    ax = fig.add_subplot(5, 50 // 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(image_t), cmap='gray')
    ax.set_title(idx2label[label_idx])
    fig.tight_layout()



class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def set_random_seed(seed_value: int):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dataloader(dataset, batch_size, pin_memory=False, shuffle=False):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dataloader


def get_y(model, device, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    loss = 0
    for (batch_input, batch_target) in dataloader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        predicted = model(batch_input)
        loss += F.nll_loss(predicted, batch_target, reduction='sum').item()  # sum up batch loss
        # get the index of the max log-probability
        _y_true = batch_target.cpu().numpy()
        _y_pred = predicted.argmax(dim=1).detach().cpu().numpy()
        y_true.append(_y_true)
        y_pred.append(_y_pred)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return loss, y_true, y_pred


def get_eval_metrics(y_true, y_pred):
    return {
      'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
      'f1': f1_score(y_true=y_true, y_pred=y_pred, average='weighted'),
    }


def get_metrics(y_true, y_pred, prefix, loss=None):
    metrics_to_log = {}
    if loss is not None:
        metrics_to_log[f"{prefix}/loss"] = loss
    metrics = get_eval_metrics(y_true=y_true, y_pred=y_pred)
    for k, v in metrics.items():
        metrics_to_log[f'{prefix}/{k}'] = v
    return metrics_to_log


def get_plots(y_true, y_pred, labels=None):
    plt.clf()
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    ax = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    ax.figure.tight_layout()
    report_fig = plt.gcf()
    plt.show()
    z = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_fig = px.imshow(
        z,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted Label", y="True Label", color="Productivity"),
        x=labels,
        y=labels,
        width=600,
        height=600
    )
    cm_fig.show()
    return report_fig, cm_fig


  
def get_images(dataset, sample, model, device, prefix):
    images = {}
    dataset = Subset(dataset, sample)
    _, y_true, y_pred = get_y(model=model, device=device, dataloader=make_dataloader(dataset, batch_size=1000))
    for sample_no, (image_t, _), actual_idx, prediction_idx  in zip(sample, dataset, y_true, y_pred):
        images[f"{prefix}_{sample_no}"] = mlf.Image(
            data_or_path=image_t.squeeze().numpy(),
            caption=f"{prefix}_{sample_no}",
            class_groups={"actuals": idx2label[actual_idx], "predictions": idx2label[prediction_idx]}
        )
    return images


# ## Log training hyperparameters

# In[14]:


args = SimpleNamespace(
    batch_size=64,
    test_batch_size=1000,
    epochs=1,
    lr=1.0,
    gamma=0.7,
    no_cuda=False,
    seed=1,
    log_interval=100,
    save_model=True
)

run.log_params(vars(args))


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_classes = len(idx2label)
set_random_seed(args.seed)


# In[16]:


# Make train and test dataloaders
train_dataloader = make_dataloader(
      train_dataset,
      batch_size=args.batch_size, 
      pin_memory=use_cuda, 
      shuffle=True
)
train_dataloader_for_eval = make_dataloader(
      train_dataset,
      batch_size=args.test_batch_size, 
      pin_memory=use_cuda, 
      shuffle=False
)
test_dataloader = make_dataloader(
      test_dataset,
      batch_size=args.test_batch_size, 
      pin_memory=use_cuda, 
      shuffle=False
)


# Initialize model and loss
model = Net(num_classes=num_classes)
criterion = torch.nn.NLLLoss()


total_steps = args.epochs * len(train_dataloader)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
model = model.to(device)
global_step = 0



for epoch in range(1, args.epochs + 1):
    epoch_start_time = timer()
    epoch_loss = torch.tensor(0.0).to(device)
    for _step, (batch_input, batch_target) in enumerate(train_dataloader):
        model.train()
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        batch_predicted = model(batch_input)
        loss = criterion(batch_predicted, batch_target)        
        
        loss.backward()
        epoch_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        model.zero_grad()

        global_step += 1
        
        if global_step % args.log_interval == 0:
            #################### Logging Metrics ###############################
            step_metrics = {
                'step/lr': scheduler.get_last_lr()[0],
                'train/step/loss': loss.item(),
            }
            _, y_true_train, y_pred_train = get_y(model, device, train_dataloader_for_eval)
            step_metrics.update(get_metrics(y_true_train, y_pred_train, prefix="train/step"))
            test_loss, y_true_test, y_pred_test = get_y(model, device, test_dataloader)
            step_metrics.update(get_metrics(y_true_test, y_pred_test, prefix="test/step", loss=test_loss))

            print(f'epoch={epoch} step={global_step}', step_metrics)
            run.log_metrics(step_metrics, step=global_step)
    
    scheduler.step()


    ###################### Logging Metrics #####################################
    epoch_loss = epoch_loss.item() / len(train_dataloader)
    epoch_time = timer() - epoch_start_time
    epoch_metrics = {
        'epoch/epoch': epoch,
        'epoch/lr': scheduler.get_last_lr()[0],
        'train/epoch/loss': epoch_loss,
        'epoch/time': epoch_time
    }
    _, y_true_train, y_pred_train = get_y(model, device, train_dataloader_for_eval)
    epoch_metrics.update(get_metrics(y_true_train, y_pred_train, prefix="train/epoch"))
    test_loss, y_true_test, y_pred_test = get_y(model, device, test_dataloader)
    epoch_metrics.update(get_metrics(y_true_test, y_pred_test, prefix="test/epoch", loss=test_loss))
    print(f'epoch={epoch} step={global_step}', epoch_metrics)

    run.log_metrics(epoch_metrics, step=global_step)


    ###################### Logging Plots #######################################
    train_report_plt, train_cm_plt = get_plots(y_true=y_true_train, y_pred=y_pred_train, labels=idx2label)
    test_report_plt, test_cm_plt = get_plots(y_true=y_true_test, y_pred=y_pred_test, labels=idx2label)
    plots = {
        'train_report': train_report_plt,
        'train_confusion_matrix': train_cm_plt,
        'test_report': test_report_plt,
        'test_confusion_matrix': test_cm_plt,
    }

    run.log_plots(plots, step=global_step)
    

    ###################### Logging Images ######################################
    train_images = get_images(dataset=train_dataset, sample=train_sample, model=model, device=device, prefix="train")
    test_images = get_images(dataset=test_dataset, sample=test_sample, model=model, device=device, prefix="test")
    images = {**train_images, **test_images}

    run.log_images(images, step=global_step)


run.end()





