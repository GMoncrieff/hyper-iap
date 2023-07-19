import argparse
from typing import Tuple
import lightning as pl
from hyperiap.datasets.point_module import PointDataModule
from hyperiap.datasets.s2_module import S2DataModule
import torch
from einops import rearrange
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score


from wandb.xgboost import WandbCallback
import wandb

# for reproducibility
pl.seed_everything(1234)


# assuming train_loader and val_loader are your PyTorch DataLoaders
def dataloader_to_numpy(
    dataloader: torch.utils.data.DataLoader, dstype: str = "all"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert data from a PyTorch DataLoader to numpy arrays.

    This function iterates over all batches from a given DataLoader,
    transforms features and labels from torch tensors to numpy arrays, and
    concatenates them. It also allows for certain modifications to the feature arrays
    based on the `dstype` parameter.

    Parameters:
    dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object that
        yields batches of (features, labels).
    dstype (str, optional): Determines the format of the returned feature arrays.
        If 'all', all feature arrays are returned as they are.
        If 'single', only the focal pixel of the feature arrays is returned.
        If 'deriv', only the focal pixel of the feature arrays is returned, and
        the derivative of this array along its last axis is computed.
        Defaults to 'all'.

    Raises:
    ValueError: If `dstype` is not one of 'all', 'single', 'deriv'.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Two numpy arrays representing the features and
        labels from the DataLoader, respectively.

    Example:
    ```python
    from hyperiap.datasets.point_module import PointDataModule

    xmod = PointDataModule()
    xmod.setup()
    valloader = xmod.val_dataloader()
    trloader = xmod.train_dataloader()

    # Convert your dataloaders to numpy arrays
    X_train, y_train = dataloader_to_numpy(trloader,dstype="all")
    X_val, y_val = dataloader_to_numpy(valloader,dstype="all")
    ```
    """

    # Initialize lists to store features and labels
    features_list = []
    labels_list = []

    for batch in dataloader:
        # Assuming your dataloader returns a tuple of (features, labels)
        features, labels = batch
        # Convert tensor to numpy and append to the list
        features_list.append(features.numpy())
        labels_list.append(labels.numpy())

    # Concatenate all the features and labels
    features_arr = np.concatenate(features_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)

    if dstype == "all":
        features_arr = rearrange(features_arr, "s x z c -> s (x z c)")
        labels_arr = rearrange(labels_arr, "s z ->(s z)")
    elif dstype == "single":
        features_arr = features_arr[:, :, 4, :]  # just the focal pixel
        features_arr = rearrange(features_arr, "s x c -> s (x c)")
        labels_arr = rearrange(labels_arr, "s z ->(s z)")
    elif dstype == "deriv":
        features_arr = features_arr[:, :, 4, :]  # just the focal pixel
        features_arr = np.gradient(features_arr, axis=-1)
        features_arr = rearrange(features_arr, "s x c -> s (x c)")
        labels_arr = rearrange(labels_arr, "s z ->(s z)")
    else:
        raise ValueError("dstype must be all, single, or deriv")

    return features_arr, labels_arr


def main(wandb_run, loader, dstype, max_depth, eta, min_child_weight, gamma):
    """
    Main function to run the XGBoost model with specified parameters.

    This function performs the following tasks:
        - Initialize a Weights & Biases (wandb) run (if specified).
        - Load the training and validation data using the PointDataModule.
        - Convert the loaded data into numpy arrays.
        - Define the parameters for XGBoost model.
        - Train the XGBoost model.
        - Evaluate the model and print the final loss and accuracy.
        - If wandb is used, logs the loss and accuracy to the wandb run and then finish the run.

    Parameters:
    wandb_run (bool): If True, wandb logging is enabled.
    loader (str): Name of the dataloader to use for model training. S2 fr Sentinel2 or IS for Hyperspectral
    dstype (str): Type of data to use for model training. Passed to the 'dataloader_to_numpy' function.
    max_depth (int): Maximum depth of the XGBoost trees.
    eta (float): Learning rate for XGBoost.
    min_child_weight (int): Minimum sum of instance weight needed in a child for XGBoost.
    gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree for XGBoost.

    Usage:
    python run_xgb.py --wandb_run --dstype='deriv' --max_depth=10 --eta=0.3 --min_child_weight=3 --gamma=0.1

    """

    # Start a wandb run
    if wandb_run:
        wandb.init(project="hyperiap")

    # prepare dataloaders
    if loader == "S2":
        xmod = S2DataModule()
    elif loader == "IS":
        xmod = PointDataModule()
    else:
        raise ValueError("loader must be S2 or IS")

    xmod.setup()
    valloader = xmod.val_dataloader()
    trloader = xmod.train_dataloader()

    # Convert your dataloaders to numpy arrays
    X_train, y_train = dataloader_to_numpy(trloader, dstype=dstype)
    X_val, y_val = dataloader_to_numpy(valloader, dstype=dstype)

    # Specify parameters for XGBoost
    param = {
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "eta": eta,
        "gamma": gamma,
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "num_class": len(np.unique(y_train)),
        "tree_method": "hist",
    }

    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    callbacks = []
    if wandb_run:
        callbacks = [wandb.xgboost.WandbCallback(log_model=True)]

    model = xgb.train(param, dtrain, callbacks=callbacks)

    # Evaluate the model
    lloss = model.eval(dval)
    y_val_pred = model.predict(dval)
    accuracy = accuracy_score(y_val, y_val_pred)
    # f1 score
    f1 = f1_score(y_val, y_val_pred, average="micro")
    start = lloss.find("mlogloss:") + len("mlogloss:")
    number = float(lloss[start:])

    if wandb_run:
        wandb.log({"loss": number, "accuracy": accuracy, "f1": f1})
        wandb.log({"final_target": f1})
        wandb.finish()

    print(f"loss: {number}, accuracy: {accuracy}, f1: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost")
    parser.add_argument("--wandb_run", action="store_true", help="Use wandb")
    parser.add_argument(
        "--loader", type=str, default="IS", help="Dataloader to use. IS or S2"
    )
    parser.add_argument("--dstype", type=str, default="all", help="Type of data to use")
    parser.add_argument("--max_depth", type=int, default=10, help="max_depth")
    parser.add_argument("--eta", type=float, default=0.3, help="eta")
    parser.add_argument(
        "--min_child_weight", type=int, default=3, help="min_child_weight"
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma")
    args = parser.parse_args()

    main(
        args.wandb_run,
        args.loader,
        args.dstype,
        args.max_depth,
        args.eta,
        args.min_child_weight,
        args.gamma,
    )
