import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_losses, test_losses, context_length, batch_size, model_type, n_layer, n_head, n_embd,
              len_train_dataset, len_test_dataset, figs_dir='figs'):
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
    info_text = f"Context Length: {context_length}\nBatch Size: {batch_size}\nModel Type: {model_type}\nLayers: {n_layer}\nHeads: {n_head}\nEmbedding Size: {n_embd}\nTrain Dataset Size: {len_train_dataset}\nTest Dataset Size: {len_test_dataset}"
    plt.text(0.02, 0.85, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    if figs_dir:
        os.makedirs(figs_dir, exist_ok=True)
        plt.savefig(os.path.join(figs_dir, 'loss_plot_with_info.svg'), format='svg')


train_losses = [5.545003414154053, 5.269860503039783, 5.017640687242339, 4.819433363178108, 4.697008356263366,
                4.640258239794381, 4.592526091804987, 4.544212329236767, 4.451634566995162, 4.272042899192134,
                3.9993650430365455, 3.639235842077038, 3.2499484940420222, 2.8579701155046875, 2.5376725166658813,
                2.2948567444765113, 2.1482279821287227, 2.08220872622502, 2.0446164298661147, 2.016295365140408,
                1.9892526178420344, 1.9241394098800948, 1.8251642260370375, 1.6895128941234154, 1.529841894590402,
                1.369733959059172, 1.2100703195680547, 1.086090538320662, 1.006573295291466, 0.9736381874808783]
test_losses = [5.707885908526044, 5.899353459823963, 6.059818389803865, 6.169892532880916, 6.216607703719029,
               6.2473020997158315, 6.284486027651055, 6.3305285919544305, 6.4508853291356285, 6.642029052556947,
               6.939108538073163, 7.2609014289323675, 7.605891382971476, 7.948435151299765, 8.23224056598752,
               8.392276120740314, 8.479882462080134, 8.531526365945506, 8.593420361363611, 8.656461116879486,
               8.82713677162348, 9.068290643913802, 9.3518744845723, 9.685232118118641, 9.987894191298373,
               10.211805277092513, 10.426186073658078, 10.575835383215615, 10.645204765852107, 10.696235767630643]

plot_loss(train_losses, test_losses, 30, 128, 'Reward-Conditioned',
          6, 8, 128, 20225, 5580)
