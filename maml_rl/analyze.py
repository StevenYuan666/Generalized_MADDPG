import numpy as np
import matplotlib.pyplot as plt


def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def analyze():
    result = np.load("../MAML_result/training_info.npy")
    inner_result = np.load("../MAML_result/inner_returns.npy")
    x = []
    inner_x=[]
    returns = []
    inner_returns=[]
    a_loss = []
    q_loss = []
    meta_q_loss = []
    for ele in result:
        x.append(ele[0])
        returns.append(ele[1])
        # a_loss.append(ele[2])
        # q_loss.append(ele[3])
        # meta_q_loss.append(ele[4])
    for ele in inner_result:
        inner_x.append(ele[0])
        inner_returns.append(ele[1])
    returns = smooth(returns)
    inner_returns=smooth(inner_returns)
    print(inner_result)
    fig, ax1 = plt.subplots(1, 2, figsize=(12, 4))
    ax1[0].plot(x, returns)
   
    # ax1[2].plot(x, q_loss)
    # ax1[3].plot(x, meta_q_loss)
    ax1[0].set_xlabel("Meta Update iterations")
    
    # ax1[2].set_xlabel("Meta Update iterations")
    # ax1[3].set_xlabel("Meta Update iterations")
    ax1[0].set_ylabel('validation returns')
    
    # ax1[2].set_ylabel('inner critic loss')
    # ax1[3].set_ylabel('meta critic loss')

    ax1[1].plot(range(len(inner_x)), inner_returns)
    ax1[1].set_xlabel("inner Update iterations")
    ax1[1].set_ylabel('inner returns')

    plt.show()

def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def test():
    path = "/Users/stevenyuan/Documents/McGill/CPSL-Lab/Generalized_MARL/Generalized_MADDPG/model/"
    load1 = np.load(path + "simple_test_load1/returns.pkl.npy")[:40]
    load2 = np.load(path + "simple_test_load2/returns.pkl.npy")[:40]
    load3 = np.load(path + "simple_test_load3/returns.pkl.npy")[:40]
    load4 = np.load(path + "simple_test_load4/returns.pkl.npy")[:40]
    load5 = np.load(path + "simple_test_load5/returns.pkl.npy")[:40]
    unload1 = np.load(path + "simple_test_unload1/returns.pkl.npy")[:40]
    unload2 = np.load(path + "simple_test_unload2/returns.pkl.npy")[:40]
    unload3 = np.load(path + "simple_test_unload3/returns.pkl.npy")[:40]
    unload4 = np.load(path + "simple_test_unload4/returns.pkl.npy")[:40]
    unload5 = np.load(path + "simple_test_unload5/returns.pkl.npy")[:40]

    x = np.arange(40)
    load_returns = np.mean([load1, load2, load3, load4, load5], axis=0)
    unload_returns = np.mean([unload1, unload2, unload3, unload4, unload5], axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 10))
    ax1[0].plot(x, load_returns, label="load pre-trained meta model")
    ax1[0].plot(x, unload_returns, label="unload pre-trained meta model")
    ax1[1].plot(x, smooth(load_returns), label="load pre-trained meta model")
    ax1[1].plot(x, smooth(unload_returns), label="unload pre-trained meta model")
    ax1[0].set_xlabel("episode * 10")
    ax1[1].set_xlabel("episode * 10")
    ax1[0].set_ylabel('validation returns')
    ax1[1].set_ylabel('smoothed validation returns')
    ax1[0].legend(loc='lower right')
    ax1[1].legend(loc='lower right')

    _, _, bars1 = ax2[0].errorbar(x, load_returns, xerr=None,
                                  yerr=np.std(load_returns, axis=0), label="load pre-trained meta model")
    _, _, bars2 = ax2[0].errorbar(x, unload_returns, xerr=None,
                                  yerr=np.std(unload_returns, axis=0), label="unload pre-trained meta model")
    [bar.set_alpha(0.3) for bar in bars1]
    [bar.set_alpha(0.3) for bar in bars2]

    _, _, bars3 = ax2[1].errorbar(x, smooth(load_returns), xerr=None,
                                  yerr=np.std(smooth(load_returns), axis=0), label="load pre-trained meta model")
    _, _, bars4 = ax2[1].errorbar(x, smooth(unload_returns), xerr=None,
                                  yerr=np.std(smooth(unload_returns), axis=0), label="unload pre-trained meta model")
    [bar.set_alpha(0.3) for bar in bars3]
    [bar.set_alpha(0.3) for bar in bars4]
    ax2[0].set_xlabel("episode * 10")
    ax2[1].set_xlabel("episode * 10")
    ax2[0].set_ylabel('validation returns')
    ax2[1].set_ylabel('smoothed validation returns')
    ax2[0].legend(loc='lower right')
    ax2[1].legend(loc='lower right')

    fig.suptitle("Averaged on five runs on test scenario", fontsize=16)
    plt.show()


if __name__ == "__main__":
    test()
