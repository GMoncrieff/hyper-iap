wandb sweep sweep.yaml --project hyperiap --verbose
wandb agent --project hyperiap --entity glennwithtwons --count=3 0xl8fvvq
wandb sweep --stop glennwithtwons/hyperiap/0xl8fvvq