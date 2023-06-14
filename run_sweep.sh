wandb sweep sweep_xgboost.yaml --project hyperiap --verbose
wandb agent --project hyperiap --entity glennwithtwons --count=50 il0fmnd0
wandb sweep --stop glennwithtwons/hyperiap/il0fmnd0