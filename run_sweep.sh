wandb sweep sweep_vit_best.yaml --project hyperiap --verbose
wandb agent --project hyperiap --entity glennwithtwons --count=1 bfnf0v3t
wandb sweep --stop glennwithtwons/hyperiap/bfnf0v3t