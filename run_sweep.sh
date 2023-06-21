wandb sweep sweep_vit.yaml --project hyperiap --verbose
wandb agent --project hyperiap --entity glennwithtwons --count=10 ki2bw88l
wandb sweep --stop glennwithtwons/hyperiap/ki2bw88l