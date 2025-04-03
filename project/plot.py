import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name, time=True):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    if time:
        ax.set_ylabel('GPT2 Execution Time (Second)')
    else:
        ax.set_ylabel('GPT2 Throughput (Tokens per Second)')

    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    # numbers for 1.3
    single_mean, single_std = 48.32030391693115, 0.10672008193355603
    device0_mean, device0_std =  25.796898102760316, 0.09330863117878814
    device1_mean, device1_std =  26.575935506820677, 2.440059298793429

    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn_time.png')

    # numbers for 1.3
    single_mean, single_std = 84423.20771596926, 216.05110396154578
    double_mean =  82704.49765262868 + 81232.77585560769
    double_std = (465.1111106105461 ** 2 + 509.54437648929604 ** 2) ** 0.5

    plot([double_mean, single_mean],
        [double_std, single_std],
        ['Data Parallel - 2 GPUs', 'Single GPU'],
        'ddp_vs_rn_tp.png',
         time=False)


    # numbers for 2.3
    # MODEL PARALLEL
    # Training time: avg:15.72153913974762, std:0.11819922924041748
    # tokens_per_second: avg: 40710.78360959045, std:306.0758365738657
    pp_mean, pp_std = 40710.78360959045, 306.0758365738657
    mp_mean, mp_std = 15.72153913974762, 0.11819922924041748
    # plot([pp_mean, mp_mean],
    #     [pp_std, mp_std],
    #     ['Pipeline Parallel', 'Model Parallel'],
    #     'pp_vs_mp.png')