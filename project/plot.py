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
    # Training time: avg:52.60557687282562, std:0.1252959966659546,
    # tokens_per_second: avg: 12166.079506576945, std:28.97717595567792

    # PIPELINE PARALLEL
    # Training time: avg:52.573574900627136, std:0.14517104625701904,
    # tokens_per_second: avg: 12173.508860297825, std:33.61462524108174
    mp_mean, mp_std = 52.60557687282562, 0.1252959966659546
    pp_mean, pp_std = 52.573574900627136, 0.14517104625701904
    mp_mean_tp, mp_std_tp = 12166.079506576945, 28.97717595567792
    pp_mean_tp, pp_std_tp = 12173.508860297825, 33.61462524108174
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp_time.png')


    plot([pp_mean_tp, mp_mean_tp],
         [pp_std_tp, mp_std_tp],
         ['Pipeline Parallel', 'Model Parallel'],
         'pp_vs_mp_tp.png',
         time=False)