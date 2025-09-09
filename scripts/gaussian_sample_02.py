import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as st

# 2D Gaussian with non-diagonal covariance (positive semidefinite)
mean_2d = np.array([0, 0])
cov_2d = np.array([[2, 1],
                   [1, 1]])  # valid covariance (symmetric PSD)
N = 1000
samples_2d = np.random.multivariate_normal(mean_2d, cov_2d, N)

def plot_confidence_ellipse(ax, mean, cov, conf=0.95, **kwargs):
    chi2_val = st.chi2.ppf(conf, df=2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:,order]
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# Plot
plt.figure(figsize=(6,6))
plt.scatter(samples_2d[:,0], samples_2d[:,1], alpha=0.4, s=10)

plot_confidence_ellipse(plt.gca(), mean_2d, cov_2d, conf=0.95, edgecolor='red', facecolor='none', lw=2, label="95% ellipse")
plot_confidence_ellipse(plt.gca(), mean_2d, cov_2d, conf=0.98, edgecolor='green', facecolor='none', lw=2, label="98% ellipse")

plt.axis("equal")
plt.title("2D Gaussian with Correlated Covariance")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
