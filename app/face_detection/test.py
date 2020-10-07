
from data_manager import FaceTrainSet
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import x1y1x2y2, non_max_suppression
import tqdm
import argparse
import yaml
from terminaltables import AsciiTable


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='config.yml', help="configuration file in yml format")
opt = parser.parse_args()

config_file = opt.config
try: 
    with open ('config.yml', 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("unable to read config file")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate(model, img_path, labels_path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    labels = []
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
    dataset = FaceTrainSet(img_path, labels_path, transforms = None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for batch_i, (imgs, targets, num_objs) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        with torch.no_grad():
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            predictions, loss = model(imgs, targets)
            if batch_i == len(dataloader)-1:
                log_str = log_metrics(model, loss, metrics, len(dataloader), train=False)
                print(log_str)
            #outputs = non_max_suppression(predictions, conf_thres, nms_thres)
        
        #sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

def log_metrics(model, loss, metrics, len_dataloader, train=False, batch_i=None, epoch=None):
    metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
    if train:
        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, config['train']['params']["epochs"], batch_i, len(dataloader))
    else:
        log_str = "\n--- Validation Metrics ---"
    for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

    log_str += AsciiTable(metric_table).table
    log_str += f"\nTotal loss {loss.item()}"
    return log_str
    

