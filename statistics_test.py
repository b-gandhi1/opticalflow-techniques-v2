from scipy.stats import pearsonr, spearmanr, weightedtau, kendalltau
import sklearn.metrics as metrics
import numpy as np
from matplotlib import pyplot as plt

x_sin = np.sin(np.linspace(0, 2 * np.pi, 100))
y_cos = (np.sin(np.linspace(0, 3 * np.pi, 100)))

corr_linear, _ = pearsonr(x_sin, y_cos) # pearson correlation (linear). Low corr ~= 0. -0.5 < poor corr < 0.5. -1 < corr_range < 1. 
corr_nonlin, _ = spearmanr(x_sin, y_cos) # spearman correlation (nonlinear). High corr ~= 1. -1 < corr_range < 1. same as pearson.
r_sq = metrics.r2_score(y_cos,x_sin)
corr_kendalltau = kendalltau(x_sin,y_cos)
corr_weightedtau = weightedtau(x_sin,y_cos, rank=True, weigher=None, additive=False)

plt.figure()
plt.subplot(211)
plt.plot(x_sin)
plt.plot(y_cos)
plt.subplot(212)
plt.plot(x_sin, y_cos, 'o')

plt.suptitle(['Pearson corr:', corr_linear, 'Spearman corr:', corr_nonlin, 'R^2:', r_sq])
print(corr_linear, corr_nonlin, r_sq, corr_kendalltau[1], corr_weightedtau[0])
plt.show()
