from models import *
#from utils.logger import *
#from utils.utils import *
from utils import *
from data_manager import *
from parse_config import *
#from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import yaml

import torch
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='config.yml', help="configuration file in yml format")
opt = parser.parse_args()

config_file = opt.config
try: 
    with open ('config.yml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("unable to read config file")

model_config = config['train']['model']
data_config = config['train']['data']
train_params = config['train']['params']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints4", exist_ok=True)

#writer = SummaryWriter()

# Get data configuration

train_labels = data_config["train_labels"]
valid_labels = data_config["valid_labels"]
img_path = data_config["img_path"]
#class_names = load_classes(data_config["class_names"])

# Initiate model
model = Darknet(model_config["model_def"]).to(device)
model.apply(weights_init_normal)

# If specified we start from checkpoint
if model_config['pretrained_weights']:
    if model_config['pretrained_weights'].endswith(".pth"):
        model.load_state_dict(torch.load(model_config["pretrained_weights"]))
    else:
        model.load_darknet_weights(model_config["pretrained_weights"])

# Get dataloader
dataset = FaceTrainSet(img_path, train_labels)
dataloader = torch.utils.data.DataLoader(dataset, train_params["batch_size"], shuffle=True,num_workers=train_params["n_cpu"] ,
            collate_fn = dataset.collate_fn)
optimizer = torch.optim.Adam(model.parameters())

metrics = [
    "grid_size",
    "loss",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "conf_obj",
    "conf_noobj",
]

for epoch in range(train_params["epochs"]):
    model.train()
    start_time = time.time()
    for batch_i, (imgs, targets, _) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + batch_i

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets)
        loss.backward()

        if batches_done % train_params["gradient_accumulations"]:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

        # ----------------
        #   Log progress
        # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, train_params["epochs"], batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % train_params["evaluation_interval"] == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                labels_path=data_config["valid_labels"],
                img_path = data_config["img_path"],
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=train_params["img_size"],
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            writer.add_scalar('mAP',
                            AP.mean(),
                            epoch                                                                                                                                                           )

        if epoch % train_params["checkpoint_interval"] == 0:
            torch.save(model.state_dict(), f"checkpoint_custom/yolov3-tiny_ckpt_%d.pth" % epoch)
