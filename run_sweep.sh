wandb sweep sweep_vit.yaml --project hyperiap --verbose
wandb agent --project hyperiap --entity glennwithtwons --count=5 zxoc04fw
wandb sweep --stop glennwithtwons/hyperiap/zxoc04fw