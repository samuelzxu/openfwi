import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Compose

import utils
import network
import transforms as T
from new_dataset import FineFWIDataset


def visualize(model, dataset, device, output_dir, num_samples, dataset_name):
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

    # Get random sample indices
    num_total_samples = len(dataset)
    if num_samples > num_total_samples:
        print(f"Warning: Number of requested samples ({num_samples}) is greater than dataset size ({num_total_samples}).")
        num_samples = num_total_samples
    
    sample_indices = random.sample(range(num_total_samples), num_samples)

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            data, label, path = dataset[idx]
            
            # Add batch dimension and move to device
            data = data.unsqueeze(0).to(device)
            
            # Get model prediction
            prediction = model(data).squeeze(0).cpu() # Remove batch dim and move to cpu

            # Denormalize prediction and ground truth
            prediction_unnorm = prediction * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            label_unnorm = label * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            
            # Squeeze out the channel dimension for plotting
            prediction_unnorm = prediction_unnorm.squeeze(0)
            label_unnorm = label_unnorm.squeeze(0)

            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Ground Truth
            im1 = axes[0].imshow(label_unnorm, cmap='jet', vmin=1.5, vmax=4.5)
            axes[0].set_title(f'Ground Truth (Sample {idx})')
            axes[0].set_ylabel('Depth (m)', fontsize=12)
            axes[0].set_xlabel('Offset (m)', fontsize=12)
            fig.colorbar(im1, ax=axes[0], label='km/s')

            # Prediction
            im2 = axes[1].imshow(prediction_unnorm, cmap='jet', vmin=1.5, vmax=4.5)
            axes[1].set_title('Prediction')
            axes[1].set_xlabel('Offset (m)', fontsize=12)
            fig.colorbar(im2, ax=axes[1], label='km/s')

            for ax in axes:
                ax.set_xticks(range(0, 70, 10))
                ax.set_xticklabels(range(0, 700, 100))
                ax.set_yticks(range(0, 70, 10))
                ax.set_yticklabels(range(0, 700, 100))

            plt.tight_layout()
            
            # Save the figure
            base_filename = os.path.basename(path).replace('.npy', '')
            save_path = os.path.join(output_dir, f'comparison_{base_filename}_sample_{idx}.png')
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

    # Run visualization
    visualize(model, dataset_valid, device, args.output_dir, args.num_samples, args.dataset)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FWI Visualization')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    
    # Path related
    parser.add_argument('-v', '--val-anno', default='split_files/val_ds.csv', help='name of val anno')
    parser.add_argument('-o', '--output-dir', default='outputs', help='path to parent folder to save images')
    parser.add_argument('-r', '--resume', required=True, help='resume from checkpoint')

    # Model related
    parser.add_argument('-m', '--model', type=str, required=True, help='inverse model name')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    
    # Visualization related
    parser.add_argument('-n', '--num-samples', default=5, type=int, help='number of samples to visualize')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args) 