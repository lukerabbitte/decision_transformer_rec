import os
import numpy as np
import matplotlib.pyplot as plt


def get_next_filename(figs_dir, base_filename):
    i = 1
    while True:
        new_filename = f"{base_filename}_{i}.svg"
        if not os.path.exists(os.path.join(figs_dir, new_filename)):
            return new_filename
        i += 1


def plot_loss(train_losses, test_losses, context_length, batch_size, model_type, n_layer, n_head, n_embd,
              filename_train_dataset, len_train_dataset, filename_test_dataset, len_test_dataset, learning_rate, lr_decay, figs_dir='figs'):
    plt.rcParams.update({'font.family': 'monospace'})
    plt.figure(figsize=(12, 8))

    # Plot training loss
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')

    # Plot test loss
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='orange')

    plt.xlabel('Epoch', fontweight='bold', fontsize=14)
    plt.ylabel('Loss', fontweight='bold', fontsize=14)
    plt.title('Training and Test Loss Over Epochs', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Information box
    info_text = f"Context Length: {context_length}\nBatch Size: {batch_size}\nModel Type: {model_type}\nLayers: {n_layer}\nHeads: {n_head}\nEmbedding Size: {n_embd}\nTrain Dataset Name: {filename_train_dataset}\nTrain Dataset Size: {len_train_dataset}\nTest Dataset Name: {filename_test_dataset}\nTest Dataset Size: {len_test_dataset}\nLearning Rate: {learning_rate}\nLearning Rate Decay: {lr_decay}"
    plt.text(0.02, 0.85, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    if figs_dir:
        os.makedirs(figs_dir, exist_ok=True)
        base_filename = 'loss_plot_with_info'
        new_filename = get_next_filename(figs_dir, base_filename)
        plt.savefig(os.path.join(figs_dir, new_filename), format='svg')

