"""
Script to load and plot benchmark results from benchmark_cifar.py

Usage:
    python plot_benchmark_results.py --save_dir ./plots

    python plot_benchmark_results.py --save_dir imgnet_final --dataset imagenet
    
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


# Default font sizes - modify these as needed
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 'xx-small'
BAR_LEGEND_FONTSIZE = 'small'


# Color mapping for different importance criteria
colors = {
    'WHC': 'C0',         # Blue
    'L1': 'C1',          # Orange
    'FPGM': 'C2',        # Green
    'BN Scale': 'C7',    # Gray
    'Random': 'C4',      # Purple
    
    'Taylor': 'C5',      # Brown
    'Hessian': 'C6',     # Pink
    'JBR_ind': 'C9',     # Olive
    'JBR': 'C10',        # Cyan
    'JBR_diag_ind': 'C12',  # Lime
    'JBR_diag': 'C13',      # Teal
    'Jacobian': 'C3',    # Red
    'Jacobian Random Labels': 'C14',    # Violet
    'Jacobian Isolated': 'black',    # black
}


def plot_results(results_path, save_dir=None, dataset='cifar', time_record=None,
                 label_fontsize=LABEL_FONTSIZE, tick_fontsize=TICK_FONTSIZE,
                 title_fontsize=TITLE_FONTSIZE, legend_fontsize=LEGEND_FONTSIZE,
                 bar_legend_fontsize=BAR_LEGEND_FONTSIZE):
    """
    Load results from a .pth file and generate plots.
    
    Args:
        results_path: Path to the .pth file containing results
        save_dir: Directory to save plots (defaults to same directory as results_path)
        dataset: Dataset name ('cifar' or 'imagenet')
        time_record: Optional dict with timing information
        label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
        title_fontsize: Font size for titles
        legend_fontsize: Font size for legends in line plots
        bar_legend_fontsize: Font size for legends in bar plots
    """
    # Load results
    print(f'Loading results from {results_path}...')
    data = torch.load(results_path + '/results.pth', weights_only=False)
    
    iterative_steps = data['iterative_steps']
    pruning_ratio = data['pruning_ratio']
    N_batchs = data['N_batchs']
    batch_size = data['batch_size']
    params_record = data['params_record']
    macs_record = data['macs_record']
    train_loss_record = data['train_loss_record']
    train_acc_record = data['train_acc_record']
    val_loss_record = data['val_loss_record']
    val_acc_record = data['val_acc_record']
    
    # Set save directory
    if save_dir is None:
        save_dir = os.path.dirname(results_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract middle_name from filename if possible
    filename = os.path.basename(results_path)
    if filename.startswith('a_record_'):
        middle_name = filename.replace('a_record_', '').replace('.pth', '')
        # Remove network name prefix if present
        parts = middle_name.split('_', 1)
        if len(parts) > 1:
            middle_name = parts[1]
    else:
        middle_name = 'results'
    
    ######################### draw #########################
    print('Drawing figures for accuracy and loss...')
    figs = ['Train', 'Validate'] if dataset != 'imagenet' else ['Validate']
    
    for fig in figs:
        if fig == 'Train':
            acc_record, loss_record = train_acc_record, train_loss_record
        elif fig == 'Validate':
            acc_record, loss_record = val_acc_record, val_loss_record

        # Pruned proportion vs Accuracy
        plt.figure()
        for index, imp_name in enumerate(params_record.keys()):
            color = colors.get(imp_name, f'C{index}')
            plt.errorbar(np.linspace(0, pruning_ratio, iterative_steps+1)*100, 
                        np.mean(acc_record[imp_name], axis=0), 
                        yerr=np.std(acc_record[imp_name], axis=0), 
                        fmt='o', ms=4, capsize=3, color=color, linestyle='-', label=imp_name)
        plt.xlabel('Pruned Filters (%)', fontsize=label_fontsize)
        plt.ylabel(f'{fig} Accuracy', fontsize=label_fontsize)
        plt.tick_params(axis='both', labelsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize, loc='upper right', framealpha=0, frameon=False)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/Propotion_{fig}_acc_{dataset}_.pdf')
        plt.close()

        # Parameters vs Accuracy
        plt.figure()
        for index, imp_name in enumerate(params_record.keys()):
            color = colors.get(imp_name, f'C{index}')
            plt.errorbar(np.array(params_record[imp_name]).mean(axis=0), 
                        np.mean(acc_record[imp_name], axis=0), 
                        yerr=np.std(acc_record[imp_name], axis=0), 
                        fmt='o', ms=4, capsize=3, color=color, linestyle='-', label=imp_name)
        plt.xlabel('# Parameters', fontsize=label_fontsize)
        plt.ylabel(f'{fig} Accuracy', fontsize=label_fontsize)
        plt.tick_params(axis='both', labelsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0, frameon=False)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/params_{fig}_acc_{middle_name}.pdf')
        plt.close()

        # Macs vs Accuracy
        plt.figure()
        for index, imp_name in enumerate(params_record.keys()):
            color = colors.get(imp_name, f'C{index}')
            plt.errorbar(np.array(macs_record[imp_name]).mean(axis=0), 
                        np.mean(acc_record[imp_name], axis=0), 
                        yerr=np.std(acc_record[imp_name], axis=0), 
                        fmt='o', ms=4, capsize=3, color=color, linestyle='-', label=imp_name)
        plt.xlabel('# MACs', fontsize=label_fontsize)
        plt.ylabel(f'{fig} Accuracy', fontsize=label_fontsize)
        plt.tick_params(axis='both', labelsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0, frameon=False)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/macs_{fig}_acc_{middle_name}.pdf')
        plt.close()

    print(f'Plots saved to {save_dir}')