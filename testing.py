import os, sys, time, pprint, h5py, argparse, json
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from models.networks import build_model
from models.losses import get_loss_func
from config.global_var import get_class_names
from utils.load_config import get_config
from utils import misc

from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset

from data_loader.augmentation import ToTensor, ZeroPad2D, AddGaussianNoise
from data_loader import dataset as dset
from data_loader import loader

from utils.metrics import iou_score, precision_recall
from os.path import join


def setup_options():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="config\model_eval.yaml",
        type=str,
    )
    parser.add_argument("--aug", action='append', help="List of augmentations to use.", default=None)

    parser.add_argument("--opt", action='append', help="List of augmentations to use.")

    parser.add_argument(
        "opts",
        help="See FastSurferCNN/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


# Configuration File 불러오기
args = setup_options()
cfg = get_config(args)
test_loader = loader.get_dataloader(cfg, "val")

if args.aug is not None:
    cfg.DATA.AUG = args.aug

if args.opt:
    cfg.DATA.CLASS_OPTIONS = args.opt

summary_path = misc.check_path(join(cfg.LOG_DIR, 'summary'))
if cfg.EXPR_NUM == "Default":
    cfg.EXPR_NUM = str(misc.find_latest_experiment(join(cfg.LOG_DIR, 'summary')) + 1)

if cfg.TRAIN.RESUME and cfg.TRAIN.RESUME_EXPR_NUM != "Default":
    cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM

cfg.SUMMARY_PATH = misc.check_path(join(summary_path, '{}'.format(cfg.EXPR_NUM)))
cfg.CONFIG_LOG_PATH = misc.check_path(join(cfg.LOG_DIR, "config", '{}'.format(cfg.EXPR_NUM)))

with open(join(cfg.CONFIG_LOG_PATH, "config.yaml"), "w") as json_file:
    json.dump(cfg, json_file, indent=2)

"""
Model 불러와서 가져온 Test data -> hdf5 변환 상태로 결과값 출력하기 
"""
model = build_model(cfg)
device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device_)

if cfg.NUM_GPUS > 1:
        assert cfg.NUM_GPUS <= torch.cuda.device_count(), \
            "Cannot use more GPU devices than available"
        print("Using ", cfg.NUM_GPUS, "GPUs!")
        model = torch.nn.DataParallel(model)

model = model.to(device_)
pic_path = "checkpoints/aparc_vinn_axial_v2.0.0.pkl"
model_state = torch.load(pic_path, map_location=device_) 
model.load_state_dict(model_state['model_state'])
class_names = get_class_names(cfg.DATA.PLANE, cfg.DATA.CLASS_OPTIONS)
num_classes = cfg.MODEL.NUM_CLASSES


""" 메인 """
val_loss_total = defaultdict(float)
val_loss_dice = defaultdict(float)
val_loss_ce = defaultdict(float)

ints_ = defaultdict(lambda: np.zeros(num_classes - 1))
unis_ = defaultdict(lambda: np.zeros(num_classes - 1))
miou = np.zeros(num_classes - 1)
per_cls_counts_gt = defaultdict(lambda: np.zeros(num_classes - 1))
per_cls_counts_pred = defaultdict(lambda: np.zeros(num_classes - 1))
accs = defaultdict(lambda: np.zeros(num_classes - 1))  # -1 to exclude background (still included in val loss)
loss_func = get_loss_func(cfg)
val_start = time.time()

for curr_iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

    images, labels, weights, scale_factors = batch['image'].to(device_), \
                                                batch['label'].to(device_), \
                                                batch['weight'].float().to(device_), \
                                                batch['scale_factor']

    pred = model(images, scale_factors)
    loss_total, loss_dice, loss_ce = loss_func(pred, labels, weights)

    sf = torch.unique(scale_factors)
    if len(sf) == 1:
        sf = sf.item()
        val_loss_total[sf] += loss_total.item()
        val_loss_dice[sf] += loss_dice.item()
        val_loss_ce[sf] += loss_ce.item()

        _, batch_output = torch.max(pred, dim=1)

        # Calculate iou_scores, accuracy and dice confusion matrix + sum over previous batches
        int_, uni_ = iou_score(batch_output, labels, num_classes)
        ints_[sf] += int_
        unis_[sf] += uni_

        tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels, num_classes)
        accs[sf] += tpos
        per_cls_counts_gt[sf] += pcc_gt
        per_cls_counts_pred[sf] += pcc_pred

    plot_dir = "./output"
    # Plot sample predictions
    if curr_iter == (len(test_loader) // 2):
        file_save_name = os.path.join(plot_dir+'Test_Predictions.pdf')
        plt_title = "dddd" 
        misc.plot_predictions(images, labels, batch_output, plt_title, file_save_name) 

for key in accs.keys():
    ious = ints_[key] / unis_[key]
    miou += ious
    val_loss_total[key] /= (curr_iter + 1)
    val_loss_dice[key] /= (curr_iter + 1)
    val_loss_ce[key] /= (curr_iter + 1)

    print("SF: {}, MIoU: {:.4f}; "
                            "Mean Recall: {:.4f}; "
                            "Mean Precision: {:.4f}; "
                            "Avg loss total: {:.4f}; "
                            "Avg loss dice: {:.4f}; "
                            "Avg loss ce: {:.4f}".format(key, np.mean(ious),
                                                        np.mean(accs[key] / per_cls_counts_gt[key]),
                                                        np.mean(accs[key] / per_cls_counts_pred[key]),
                                                        val_loss_total[key], val_loss_dice[key], val_loss_ce[key]))