"""
Dataset paths.
"""
"""Dataset paths used by the training and inference dataloaders.

Keep these paths local to the checkout so the repository works after cloning
on another machine.  Override ``LMAFORMER_DATASET_ROOT`` when the dataset is
stored elsewhere.
"""
import os


_dataset_root = os.environ.get(
    "LMAFORMER_DATASET_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Datasets", "IRDST")),
)
train_seqs_file = os.path.join(_dataset_root, "ImageSets", "train_new.txt")
_val_split = os.environ.get("LMAFORMER_VAL_SPLIT", "val")
val_seqs_file = os.path.join(_dataset_root, "ImageSets", "{}_new.txt".format(_val_split))
img_path = os.path.join(_dataset_root, "images")
mask_path = os.path.join(_dataset_root, "masks")


