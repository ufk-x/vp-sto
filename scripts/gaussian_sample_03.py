import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as st

# Parameters of correlated Gaussian
mean = np.array([0, 0])
cov = np.array([[2, 1],
                [1, 1]])  # correlated covariance
N = 1000

# ----- Direct sampling -----
samples_direct = np.random.multivariate_normal(mean, cov, N)

# ----- Whitened sampling -----
z = np.random.randn(N, 2)  # N(0, I)
L = np.linalg.cholesky(cov)  # cov = L L^T
samples_whitened = mean + z @ L.T

# --- helper: draw confidence ellipse (conf in (0,1)) ---
def add_confidence_ellipse(ax, mean, cov, conf=0.95, **kwargs):
    chi2_val = st.chi2.ppf(conf, df=2)           # radius^2 in Mahalanobis metric
    eigvals, eigvecs = np.linalg.eigh(cov)       # eigen-decomposition (symmetric PSD)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    width, height = 2 * np.sqrt(eigvals * chi2_val)  # full axis lengths
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# -------- Figure 1: Direct sampling + confidence ellipses --------
plt.figure()
plt.scatter(samples_direct[:, 0], samples_direct[:, 1], alpha=0.4, s=10)
add_confidence_ellipse(plt.gca(), mean, cov, conf=0.95, edgecolor='red', facecolor='none', lw=2, label='95% ellipse')
add_confidence_ellipse(plt.gca(), mean, cov, conf=0.98, edgecolor='green', facecolor='none', lw=2, label='98% ellipse')
plt.axis('equal')
plt.title("Direct Sampling from N(mean, cov) with Confidence Ellipses")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()


# -------- Figure 2: Whitened sampling + confidence ellipses --------
plt.figure()
plt.scatter(samples_whitened[:, 0], samples_whitened[:, 1], alpha=0.4, s=10, color='orange')
add_confidence_ellipse(plt.gca(), mean, cov, conf=0.95, edgecolor='red', facecolor='none', lw=2, label='95% ellipse')
add_confidence_ellipse(plt.gca(), mean, cov, conf=0.98, edgecolor='green', facecolor='none', lw=2, label='98% ellipse')
plt.axis('equal')
plt.title("Whitened Sampling via z @ L.T with Confidence Ellipses")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
