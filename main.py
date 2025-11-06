import torch 
import ray
from ray import tune 
from ray.tune.schedulers import ASHAScheduler
import torch.nn as nn 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import os
import torch.optim as optim 
from ray import train

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_object_spilling_config"] = '{"type":"filesystem","params":{"directory_path":"/tmp/spill"}}'
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

ray.init(
    num_cpus=2,
    object_store_memory=500*1024*1024,
    _temp_dir='/tmp/ray',
    ignore_reinit_error=True
)


def train(config):
    train_loader = DataLoader(datasets.MNIST(
        root='data', train=True, download=True, transform=transforms.ToTensor()), 
                              batch_size=64, shuffle=True,
                              num_workers=0,
                              pin_memory=False
    )
    val_loader = DataLoader(datasets.MNIST(
        root='data', train=False, transform=transforms.ToTensor()),
                            batch_size=64, shuffle=True,
                            num_workers=0,
                            pin_memory=False
    )

    layers = []
    in_channels=1
    current_size=28 

    for _ in range(config['num_layers']):
        
        kernel_size = config['kernel_size']
        if current_size<kernel_size:
            break 

        layers +=[
            nn.Conv2d(in_channels, config['hidden_dim'], kernel_size, padding=kernel_size//2),
            getattr(nn, config['activation'])(),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout'])
        ]
        in_channels = config['hidden_dim']
        current_size=current_size//2 
    final_size = current_size

    model = nn.Sequential(*layers, nn.Flatten(), nn.Linear(in_channels*final_size*final_size, 10))
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss= criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x).argmax(1)
                correct += (out == y).sum().item()
                total += y.size(0)
        acc = correct/total 
        tune.report({'accuracy':acc})
    del model, optimizer, train_loader, val_loader


search_space = {
    "num_layers": tune.randint(2, 6),
    "hidden_dim": tune.choice([32, 64, 128, 256]),
    "kernel_size": tune.choice([3, 5]),
    "activation": tune.choice(["ReLU", "GELU", "SiLU"]),
    "dropout": tune.uniform(0.1, 0.5),
    "lr": tune.loguniform(1e-4, 1e-2),
}

scheduler = ASHAScheduler(
    metric='accuracy',
    mode='max',
    grace_period=1,
    reduction_factor=2 
)

config = tune.TuneConfig(
    scheduler=scheduler,
    num_samples=20,
    max_concurrent_trials=2,
)

path = f"file://{os.path.abspath('results')}"

tuner = tune.Tuner(
    tune.with_resources(train, {"cpu":1}),
    tune_config=config,
    param_space=search_space,
    run_config=tune.RunConfig(
        storage_path=path, name='testing',
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=False
        ),
        verbose=1,
    ),

)

results = tuner.fit()
best_result = results.get_best_result(metric="accuracy", mode="max")
print("Best config:", best_result.config)
print("Best accuracy:", best_result.metrics["accuracy"])

ray.shutdown()
