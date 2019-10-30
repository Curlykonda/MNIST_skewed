import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_train_test(res_dir, loc="best", show=False):
    stats = pickle.load(open(res_dir + "/stats.pkl", "rb"))

    #(exp_dirs, labels, metrics_to_exclude, n_epochs=10, format_lbl=True, loc='best', ncol=1):
    second_plot = ['loss']

    plt.style.use('seaborn')

    fig = plt.figure(figsize=(8, 9))

    ax1 = fig.add_subplot(2, 1, 1)  # xticklabels=[], yticklabels=[], xticks=[], yticks=[], fc="red",
    ax2 = fig.add_subplot(2, 1, 2)
    # n_colors = (len(all_metrics)-len(metrics_to_exclude))*len(exp_dirs)
    n_colors = len(stats) / 2
    colors = [plt.cm.jet(x) for x in np.linspace(0., 1., n_colors)]
    color_counter = 0

    # Aggregate metrics in dict
    plot_metrics = {'train': {}, 'test': {}}

    for key, values in stats.items():
        if key.__contains__("train"):
            plot_metrics['train'][key.split("_")[-1]] = values
        else:
            plot_metrics['test'][key.split("_")[-1]] = values



        #if format_lbl:
        #    lbl = ("p:%.1f, T:%d" % (labels[i_exp][0], labels[i_exp][1]))

    for i, (key, values) in enumerate(plot_metrics['train'].items()):
        color = colors[color_counter]

        if key not in second_plot:
            cur_ax = ax1
        else:
            cur_ax = ax2

        # train
        cur_ax.plot(values, label='{} (train)'.format(key), color=color, linestyle='dashed')

        # test
        cur_ax.plot(plot_metrics['test'][key], label='{} (test)'.format(key), color=color, linestyle='solid')

        # cur_ax.legend(loc='lower left', bbox_to_anchor=(1.04, 0))
        # cur_ax.legend(loc='best', ncol=2)
        # loc='upper center', bbox_to_anchor=(0.5, -0.05),

        color_counter += 1

    ax1.legend(loc=loc, fancybox=True, shadow=True, ncol=2)
    ax2.legend(loc=loc, fancybox=True, shadow=True, ncol=1)
    ax1.set_title("Accuracy & F1 Score")
    ax2.set_title("Loss")

    plt.xlabel('Epoch')
    plt.savefig(res_dir + '/performance.png', dpi=200)
    if show:
        plt.show()

def show_class_distribution(targets, normalize=False, show=False):
    # targets = [target for (_, target) in trainset]
    hist, bins = np.histogram(targets, bins=len(np.unique(targets)))

    if show:
        for i in range(len(hist)):
            print("Digit: {0}  Train-Examples: {1} ({2} %)".format(i + 1, hist[i],
                                                                   np.round(hist[i] / len(targets), decimals=3) * 100))

    if normalize:
        hist = [np.round(val / len(targets), decimals=3) for val in hist]

    return hist

def plot_data_distribution(res_dir, targets):
    fig = plt.figure(figsize=(8, 9))
    plt.bar(range(10), show_class_distribution(targets, normalize=True))
    plt.title("Class distribution of MNIST digits")
    plt.xticks(np.arange(0, 10, 1))
    plt.savefig(res_dir + "/class_distribution.png", dpi=200)
    plt.show()


def main(args):
    plot_train_test(args.res_dir, show=True)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default="../results/Exp_1_False_normal_NoAug_42",
                        help='Directory for storing results')

    args, unparsed = parser.parse_known_args()

    main(args)
