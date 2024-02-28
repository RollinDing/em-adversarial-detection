import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import seaborn as sns
import pandas as pd
x = np.arange(0, 10)
Fscore1 =[0.88158729, 0.977135981,0.846286107,0.934820621,0.861755472, 0.86298110,0.843684097,0.910262678, 0.931557923, 0.914698922]
Fscore2 = [0.83182, 0.97084, 0.984341, 0.960498, 0.82, 0.99455, 0.79913, 0.998748, 0.932844, 0.994974]
Fscore3 = [0.8135,0.9469,0.948751,0.943445857,0.91,0.94086496,0.746329805,0.9492119,0.9134453, 0.948751]
Fscore4 = [0.7156,	0.9552845,	0.97833,	0.951602,0.718534,0.93328,0.70853,0.998748,0.895,0.995]
df = {'NIC':Fscore1, 'EM-VAE':Fscore2}
df = {
    "Class": [0, 1, 2, 3, 4 , 5, 6, 7, 8, 9]*4,
    "Fscore":Fscore2 +Fscore3+Fscore4+Fscore1,
    "Method": ["EM-VAE"]*10+["EM-SVM"]*10+["EM-AE"]*10+['NIC']*10
}
df =pd.DataFrame(data=df)
# print(df)

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(10, 6))
bar=sns.barplot(x="Class", y="Fscore", hue="Method", data=df)

# hatches = ['-', '+', 'x', '^']
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i%4])
#     print(hatches[i%4])


plt.setp(bar.get_legend().get_texts(), fontsize='22')
plt.setp(bar.get_legend().get_title(), fontsize='32')
bar.set_xlabel("Class", fontsize=20)
bar.set_ylabel("F-score", fontsize=20)
plt.savefig("imgs/comparison.png")