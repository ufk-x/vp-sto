import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as st

# 1D Gaussian sampling (mean=0, cov=1)
N = 1000
samples_1d = np.random.randn(N)  # standard normal

# 2D Gaussian sampling (mean=[0,0], cov=I)
mean_2d = np.array([0, 0])
cov_2d = np.eye(2)
samples_2d = np.random.multivariate_normal(mean_2d, cov_2d, N)

# Plot 1D histogram
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(samples_1d, bins=30, density=True, alpha=0.6, color='blue')
x = np.linspace(-4,4,200)
plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2), 'r-', lw=2, label="PDF N(0,1)")
plt.title("1D Gaussian Samples (N=1000)")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

# Function to draw confidence ellipse for 2D Gaussian
def plot_confidence_ellipse(ax, mean, cov, conf=0.95, **kwargs):
    # chi2 value for confidence level
    chi2_val = st.chi2.ppf(conf, df=2)
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:,order]
    # width, height (2*sqrt for full axis length)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    # angle of ellipse (rotation)
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# Plot 2D scatter with confidence ellipses
ax2 = plt.subplot(1,2,2)
ax2.scatter(samples_2d[:,0], samples_2d[:,1], alpha=0.4, s=10)

# Draw ellipses for 95% and 98% confidence
plot_confidence_ellipse(ax2, mean_2d, cov_2d, conf=0.95, edgecolor='red', facecolor='none', lw=2, label="95% ellipse")
plot_confidence_ellipse(ax2, mean_2d, cov_2d, conf=0.98, edgecolor='green', facecolor='none', lw=2, label="98% ellipse")

ax2.axis("equal")
ax2.set_title("2D Gaussian Samples (N=1000)")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.legend()

plt.tight_layout()
plt.show()
