import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, a, b, figsize=[20, 12]) -> None:
        """
        a: figure shape[0]
        b: figure shape[1]
        """
        self.fig, self.axs = plt.subplots(a, b, figsize=figsize)
        pass
    
    def drawPlot(self, x, xaxis, yaxis):
        """
        Draw line x on axis
        x: data
        axis: axis index
        """
        # self.axs[xaxis, yaxis].plot(x)
        # self.axs[xaxis, yaxis].grid(True)
        self.axs[yaxis].plot(x)
        self.axs[yaxis].grid(True)
    
    def drawImage(self, mat, xaxis, yaxis):
        """
        Draw matrix on axis
        mat: matrix
        axis: axis index
        """
        self.axs[xaxis, yaxis].imshow(mat)
    
    def save(self, fname):
        """
        Save the image
        """
        self.fig.tight_layout()
        self.fig.savefig(fname)
    
    def show(self):
        """
        Show the image
        """
        plt.show()

def plot_stft(Fxx_lst, labels_lst):
    plotters = []
    for Fxx, labels in zip(Fxx_lst, labels_lst):
        myPlotter = Plotter(2, 1)
        spectrum = (Fxx[labels==0].mean(axis=0) * 255).astype(np.int8)
        myPlotter.drawImage(spectrum, 0)
        spectrum = (Fxx[labels==1].mean(axis=0) * 255).astype(np.int8)
        myPlotter.drawImage(spectrum, 1)
        plotters.append(myPlotter)
    return plotters

def plot_average(ATTACK, target, axis):
    label_file = f"../data/data/fm/{ATTACK}/y_adv.npy"
    label = np.load(label_file)[:1000]
    label = np.squeeze(label)
    traces = []
    for INDEX in range(1000):
        FILE_PATH = f"{TRACE_PATH}/{ATTACK}/{BATCH}/{INDEX}.npy"
        trace = np.load(FILE_PATH).reshape(1, -1)
        traces.append(trace)
    traces = np.concatenate(traces, axis=0)
    traces = traces[label==target]
    print(traces.shape)
    myplotter.drawPlot(traces.mean(axis=0), xaxis=0, yaxis=1)


def draw_raw(ATTACK="org", target=1):
    """
    Plot raw traces for ccs conference
    """
    myplotter = Plotter(1, 1, [20, 6])
    BATCH = "6"
    label_file = f"../data/data/fm/{ATTACK}/y_adv.npy"
    label = np.load(label_file)[1:30]
    label = np.squeeze(label)
    benign_label = label
    traces = []
    for INDEX in range(1,30):
        FILE_PATH = f"{TRACE_PATH}/{ATTACK}/{BATCH}/{INDEX}.npy"
        trace = np.load(FILE_PATH).reshape(1, -1)
        traces.append(trace)
    traces = np.concatenate(traces, axis=0)
    traces = traces[label==target]
    
    # print(traces[0].shape)


    # myplotter.drawPlot(, xaxis=0, yaxis=0)
    myplotter.axs.plot(traces.mean(axis=0), lw=2, color='black')
    # myplotter.axs.plot(traces.mean(axis=0)[4000:4150], lw=3, color='red')
    myplotter.axs.set_ylim(-100, 100)
    myplotter.axs.set_axis_off()
    myplotter.save(f"imgs/{ATTACK}{target}.png")   

if __name__ == "__main__":
    print("Loading Finer Aligned Traces...")
    TRACE_PATH = "../data/data/fm_robust-20220321"
    TYPES = ["org", "targetcw"]
    print("Plotting Traces...")
    
    
    draw_raw()
    # draw_raw("cw1")
    # exit()
    
    # ATTACK = "org"
    # BATCH = "0"
    # target = 1
    # label_file = f"../data/data/fm/{ATTACK}/y_adv.npy"
    # label = np.load(label_file)[:30]
    # label = np.squeeze(label)
    # traces = []
    # for INDEX in range(30):
    #     FILE_PATH = f"{TRACE_PATH}/{ATTACK}/{BATCH}/{INDEX}.npy"
    #     trace = np.load(FILE_PATH).reshape(1, -1)
    #     traces.append(trace)
    # traces = np.concatenate(traces, axis=0)
    
    # traces = traces[label==target]
    # print(traces.shape)
    # myplotter.drawPlot(traces.mean(axis=0), xaxis=0, yaxis=1)

    # ATTACK = f"cw{target}"

    # label_file = f"../data/data/fm/{ATTACK}/y_adv.npy"
    # label = np.load(label_file)[:30]
    # label = np.squeeze(label)
    # traces = []
    # for INDEX in range(30):
    #     FILE_PATH = f"{TRACE_PATH}/{ATTACK}/{BATCH}/{INDEX}.npy"
    #     trace = np.load(FILE_PATH).reshape(1, -1)
    #     traces.append(trace)
    # traces = np.concatenate(traces, axis=0)
    # traces = traces[np.logical_and(benign_label==3,label==target)]
    # print(traces.shape)
    # myplotter.drawPlot(traces.mean(axis=0), xaxis=0, yaxis=2)

    # myplotter.save(f"imgs/benign{target}.pdf")  


    
