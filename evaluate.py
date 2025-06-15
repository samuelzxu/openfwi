import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

import utils
import network
import transforms as T
from new_dataset import FineFWIDataset

# Load colormap for velocity map visualization
rainbow_cmap = ListedColormap(np.load('rainbow256.npy'))


def evaluate_and_visualize_worst(model, dataset, device, output_dir, top_k, dataset_name, batch_size, num_workers):
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset normalization context
    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[dataset_name]
        except KeyError:
            print('Unsupported dataset.')
            return

    label_min, label_max = ctx['label_min'], ctx['label_max']

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )

    all_samples_with_loss = []
    sample_idx_counter = 0

    print("Evaluating model on the entire validation set...")
    with torch.no_grad():
        for j, (data, label, paths) in enumerate(dataloader):
            # Move data to device
            data = data.to(device)
            
            # Get model prediction and move to cpu for denormalization
            # UNet -> unsqueeze
            # CoolNet -> No unsqueeze
            # prediction = model(data).cpu().unsqueeze(1)  # Remove batch dim and squeeze channel dim if needed
            prediction = model(data).cpu()

            # Denormalize prediction and ground truth
            prediction_unnorm = prediction * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            label_unnorm = label.cpu() * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            
            # Calculate per-sample L1 loss
            losses = nn.L1Loss(reduction='none')(prediction_unnorm, label_unnorm).mean(dim=(1, 2, 3))

            for i, loss in enumerate(losses):
                all_samples_with_loss.append((loss.item(), sample_idx_counter + i, paths[i]))
            
            sample_idx_counter += len(data)
            print(f"Processed {sample_idx_counter}/{len(dataset)} samples...", end='\r')
    
    print("\nEvaluation finished.")

    # Sort by loss (descending) and get top K
    all_samples_with_loss.sort(key=lambda x: x[0], reverse=True)
    top_k_samples = all_samples_with_loss[:top_k]

    print(f"\nTop {top_k} worst performing samples:")
    for loss, idx, path in top_k_samples:
        print(f"Sample index: {idx}, Path: {path}, Un-normalized L1 Loss: {loss:.4f}")

    print("\nSaving visualizations for top K samples...")
    # Visualize the top K samples
    with torch.no_grad():
        for i, (loss, idx, path) in enumerate(top_k_samples):
            data, label, _ = dataset[idx]
            
            # Add batch dimension and move to device
            data = torch.from_numpy(data).unsqueeze(0).to(device)
            
            # Get model prediction
            prediction = model(data).squeeze(0).cpu() # Remove batch dim and move to cpu

            # Denormalize prediction and ground truth
            prediction_unnorm = prediction * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            label_unnorm = label * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            
            # Squeeze out the channel dimension for plotting and convert to numpy
            prediction_unnorm_np = np.array(prediction_unnorm.squeeze(0))
            label_unnorm_np = np.array(label_unnorm.squeeze(0))

            # Determine color range from ground truth
            vmin = label_unnorm_np.min()
            vmax = label_unnorm_np.max()

            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(15,9))
            
            # Ground Truth
            im = axes[0].matshow(label_unnorm_np, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
            axes[0].set_title(f'Ground Truth (Sample {idx})\nLoss: {loss:.4f}')
            axes[0].set_ylabel('Depth (m)', fontsize=12)
            axes[0].set_xlabel('Offset (m)', fontsize=12)

            # Prediction
            axes[1].matshow(prediction_unnorm_np, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title('Prediction')
            axes[1].set_xlabel('Offset (m)', fontsize=12)

            for ax in axes:
                ax.set_xticks(range(0, 70, 10))
                ax.set_xticklabels(range(0, 700, 100))
                ax.set_yticks(range(0, 70, 10))
                ax.set_yticklabels(range(0, 700, 100))

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label='Velocity(m/s)')
            
            # Save the figure
            base_filename = path.split('/')[-4]
            save_path = os.path.join(output_dir, f'{i}_{base_filename}_{idx}.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved visualization for sample {idx} to {save_path}")

def main(args):
    print(args)
    device = torch.device(args.device)

    # Load dataset context
    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            return

    # Data transforms
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=args.k))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])

    # Create dataset
    dataset_valid = FineFWIDataset(
        args.val_anno,
        sample_ratio=args.sample_temporal,
        transform_data=transform_data,
        transform_label=transform_label,
        expand_label_zero_dim=True,
        expand_data_zero_dim=False,
        squeeze=False,
        mode="val"
    )
    dataset_valid.set_return_path(True)

    # Create model
    if args.model not in network.model_dict:
        print('Unsupported model.')
        return
    model = network.model_dict[args.model]().to(device)

    # Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(network.replace_legacy(checkpoint['model']))
    else:
        print("A model checkpoint must be provided via the --resume argument.")
        return

    # Run evaluation and visualization
    evaluate_and_visualize_worst(model, dataset_valid, device, args.output_dir, args.top_k, args.dataset, args.batch_size, args.workers)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FWI Evaluation and Visualization of Worst Performing Samples')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='all', type=str, help='dataset name')
    
    # Path related
    parser.add_argument('-v', '--val-anno', default='split_files/val_ds.csv', help='name of val anno')
    parser.add_argument('-o', '--output-dir', default='outputs_eval', help='path to parent folder to save images')
    parser.add_argument('-r', '--resume', required=True, help='resume from checkpoint')

    # Model related
    parser.add_argument('-m', '--model', type=str, required=True, help='inverse model name')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    
    # Evaluation and Visualization related
    parser.add_argument('-t', '--top-k', default=5, type=int, help='number of worst samples to visualize')
    parser.add_argument('-k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size for evaluation')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args) 