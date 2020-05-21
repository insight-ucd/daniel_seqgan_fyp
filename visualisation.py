import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import pickle

def plotCurve(loss, legend, title):
    plt.clf()
    plt.plot(loss)
    plt.title('Learning Curve')
    plt.ylabel('NLL')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper right')
    plt.savefig(title + ".png", bbox_inches="tight", dpi=100)
    plt.show()


def plotBLEU(generations, title, legend):
    plt.clf()
    plt.plot(generations)
    plt.title(title)
    plt.ylabel('BLEU')
    plt.xlabel('Epoch')
    plt.legend(legend, loc='upper right')
    plt.savefig(title + ".png", bbox_inches="tight", dpi=100)
    plt.show()


def plotSessions(results, label, filename):
    plt.clf()
    plt.ylabel("Number of Sessions")
    plt.xlabel("Session Length (minutes)")

    cmap = cm.get_cmap('viridis')
    bins = np.linspace(0, 50, 10)

    plt.style.use('seaborn-deep')
    plt.hist([results[5], results[35]], bins, label=label)
    plt.legend(loc='upper right')
    plt.savefig(filename, dpi=100)
    plt.show()


# Plot Session Lengths for varios values of Δt
with open('all_tracks.pkl', 'rb') as handle:
    all_tracks = pickle.load(handle)
with open('results_3.pkl', 'rb') as handle:
    results = pickle.load(handle)
with open('sessions.pkl', 'rb') as handle:
    token_stream = pickle.load(handle)
plotSessions([results[5], results[35]],
             ['\u0394t = 5 minutes', '\u0394t = 35 minutes'],
             "delta5_35_histogram.png")

plotSessions([results[10], results[20], results[30]],
             ['\u0394t = 10 minutes', '\u0394t = 20 minutes', '\u0394t = 30 minutes'],
             "delta10_20_30_histogram.png")

# Plot Learning Curve for various values of D_STEPS
for D_STEPS in range(1, 5):
    with open('results_' + str(D_STEPS) + '.pkl', 'rb') as handle:
        res = pickle.load(handle)
    x = []
    for r in range(100):
        x.append(res['unsupervised_g_losses'][r])
    print("Loss for D_STEPS = {}: {}".format(D_STEPS, res['unsupervised_g_losses'][99]))

    plotCurve(x, ['D_STEPS = ' + str(D_STEPS)], "CMON_" + str(D_STEPS))

# Plot BLEU Score for each generation for various values of D_STEPS
for D_STEPS in range(1, 5):
    with open('results_' + str(D_STEPS) + '.pkl', 'rb') as handle:
        results = pickle.load(handle)
    x = []
    for epoch in range(100):
        if results['unsupervised_generations'][epoch] is not None:
            x.append(sentence_bleu(token_stream, results['unsupervised_generations'][epoch],
                                   smoothing_function=SmoothingFunction().method1))

    plotBLEU(x, "BLEU_" + str(D_STEPS), "D_STEP = " + str(D_STEPS))

# Peer versus BLEU Score Graph
x = []
with open('results_3.pkl', 'rb') as handle:
    results = pickle.load(handle)
    # print(epoch, type(results['unsupervised_generations'][epoch]))
    for epoch in range(96, 100):
        print("Epoch {}: {}".format(epoch, results['unsupervised_generations'][epoch]))
        for tr in results['unsupervised_generations'][epoch]:
            print("{} - {}".format(all_tracks[tr]['track_name'], all_tracks[tr]['artist']))
        print("{:2f}".format(sentence_bleu(token_stream, results['unsupervised_generations'][epoch],
                                           smoothing_function=SmoothingFunction().method4)))
        print()
