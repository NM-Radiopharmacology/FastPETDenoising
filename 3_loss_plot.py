"""
(optional) script to generate a matplotlib figure window with the loss plot (live update)
"""
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from functools import partial


def live_loss_plot(i, log_file_path, figure):

    # data
    losses = open(log_file_path, "r").read()
    losses = losses.split('\n')
    losses.remove('')

    loss_function = losses[0].split('(')[1].split(')')[0]

    # plot
    plt.clf()
    ax = figure.add_subplot(1, 1, 1)
    ax.clear()

    if len(losses) > 2:

        # data
        training_epochs = []
        training_losses = []

        validation_epochs = []
        validation_losses = []

        baseline_validation_loss = float(losses[1].split(',')[-1])
        best_validation_loss = baseline_validation_loss

        for e in losses[2:-1]:
            e = e.split(',')
            training_epochs.append(int(e[0]))
            training_losses.append(float(e[1]))

            if len(e) > 2:
                validation_epochs.append(int(e[0]))
                validation_losses.append(float(e[2]))

        best_validation_losses = []
        best_validation_epochs = []
        for (n, v) in zip(validation_epochs, validation_losses):
            if v < best_validation_loss:
                best_validation_loss = v
                best_validation_epochs.append(n)
                best_validation_losses.append(v)

        # plot
        # plotting the baseline validation loss
        ax.axhline(baseline_validation_loss, color='#A6ACAF', linestyle='--',
                   label="baseline validation loss")
        plt.annotate(str(np.round(baseline_validation_loss, decimals=4)),
                     (1, baseline_validation_loss), textcoords='offset pixels',
                     xytext=(8, -12), fontsize=8)

        # plotting training loss
        ax.plot(training_epochs, training_losses, color='#76D7C4', label='training loss')

        # plotting validation loss
        ax.plot(validation_epochs, validation_losses, color='#E9967A', label='validation loss')

        # plotting best validation losses
        if len(best_validation_losses) >= 1:
            ax.plot(best_validation_epochs, best_validation_losses, 'v', markersize=5, color='#A6ACAF',
                    label='Validation loss minimums')
            ax.plot(best_validation_epochs[-1], best_validation_losses[-1], 'v', color='#5A5A5A')
            plt.annotate(str(np.round(best_validation_losses[-1], decimals=6)),
                         (best_validation_epochs[-1], best_validation_losses[-1]), textcoords='offset pixels',
                         xytext=(-45, 10), fontsize=8)

        closest_multiple_of_5 = len(training_epochs)
        if closest_multiple_of_5 % 5 != 0:
            while closest_multiple_of_5 % 5 != 0:
                closest_multiple_of_5 = closest_multiple_of_5 + 1

        xticks = training_epochs

        if len(training_epochs) > 25:
            xticks_spacing = 5
            while closest_multiple_of_5 / xticks_spacing > 25:
                xticks_spacing = xticks_spacing + 5
            xticks = [i for i in range(closest_multiple_of_5+xticks_spacing)][::xticks_spacing]

        ax.set_xticks(xticks)
        ax.set_xlim(1, xticks[-1])

        if (np.median(training_losses) * 3) > np.median(validation_losses):
            ax.set_ylim(0, np.median(training_losses) * 3)
        else:
            ax.set_ylim(0, np.median(validation_losses) * 2)

    elif len(losses) == 2:
        ax.axhline(float(losses[1].split(',')[-1]), color='#A6ACAF', linestyle='--', label="baseline validation loss")

    if len(losses) < 2:
        ax.grid('ON', color='#F2F3F4')
        plt.xlabel('epoch')
        plt.ylabel(f'loss ({loss_function})')
        figure.tight_layout()

    plt.legend(loc='best')


current_training_folder = None
latest_creation = None
for item in os.listdir(os.getcwd()):
    if os.path.isdir(item) and item.startswith("training_"):
        creation = datetime.strptime(
            time.ctime(os.path.getctime(os.path.join(os.getcwd(), item))), "%a %b %d %H:%M:%S %Y")

        if latest_creation is None:
            latest_creation = creation
            current_training_folder = os.path.join(os.getcwd(), item)

        if creation > latest_creation:
            latest_creation = creation
            current_training_folder = os.path.join(os.getcwd(), item)

current_fold = 0
for item in os.listdir(current_training_folder):
    if item.startswith("training_log"):
        fold = int(item.split("fold")[-1].replace(".txt", ""))
        if fold > current_fold:
            current_fold = fold

training_log_file = os.path.join(current_training_folder, f"training_log_fold{current_fold}.txt")

fig = plt.figure(figsize=(12, 6), num=f"{current_training_folder.split(os.sep)[-1]}_fold{current_fold}_loss_plot")
anim = animation.FuncAnimation(fig,
                               partial(live_loss_plot, log_file_path=training_log_file, figure=fig),
                               interval=5000, cache_frame_data=False)

plt.show(block=True)
fig.savefig(os.path.join(current_training_folder, f"loss_plot_fold{current_fold}.png"), dpi=300)
