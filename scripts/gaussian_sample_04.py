import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as st

# ---------- Precision-form Product of Gaussians (info form) ----------
# For K Gaussians N(mu_i, Sigma_i) with precisions Lambda_i = Sigma_i^{-1}:
#   Lambda_post = sum_i Lambda_i
#   h_post      = sum_i (Lambda_i @ mu_i)
#   Sigma_post  = inv(Lambda_post)
#   mu_post     = Sigma_post @ h_post
#
# In our case we start with a standard Gaussian prior N(0, I) on x,
# and "inject" two Gaussian priors via product:
#   p(x) ∝ N(x | 0, I) * N(x | mu1, Sigma1) * N(x | mu2, Sigma2)
# => in precision form:
#   Lambda_post = I + Lambda1 + Lambda2
#   h_post      = 0 + Lambda1 mu1 + Lambda2 mu2
#   Sigma_post  = (Lambda_post)^{-1}
#   mu_post     = Sigma_post h_post

rng = np.random.default_rng(0)

# Dimensionality (2D for visualization)
d = 2
I = np.eye(d)

# ---- Base variable (standard Gaussian) ----
N = 1200
x_base = rng.standard_normal(size=(N, d))  # N(0, I)

# ---- Two Gaussian "priors" to inject ----
mu1 = np.array([1.5, -0.5])
Sigma1 = np.array([[0.6, 0.25],
                   [0.25, 0.8]])
Lambda1 = np.linalg.inv(Sigma1)

mu2 = np.array([-0.5, -1.0])
Sigma2 = np.array([[0.9, -0.2],
                   [-0.2, 0.5]])
Lambda2 = np.linalg.inv(Sigma2)

# ---- Product in precision (information) form ----
Lambda_post = I + Lambda1 + Lambda2
h_post = Lambda1 @ mu1 + Lambda2 @ mu2  # (base had mean 0, so +0)
Sigma_post = np.linalg.inv(Lambda_post)
mu_post = Sigma_post @ h_post

# ---- Sampling: posterior via Cholesky ----
L_post = np.linalg.cholesky(Sigma_post)
z = rng.standard_normal(size=(N, d))
x_post = mu_post + z @ L_post.T

# ---- Helper: draw confidence ellipse ----
def add_conf_ellipse(ax, mean, cov, conf=0.95, **kwargs):
    chi2_val = st.chi2.ppf(conf, df=2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    ell = Ellipse(mean, width, height, angle=angle, **kwargs)
    ax.add_patch(ell)

# ---- Plot: before vs after product-of-Gaussians ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left: base standard Gaussian
axes[0].scatter(x_base[:, 0], x_base[:, 1], s=10, alpha=0.35)
axes[0].set_title("Before: Standard Gaussian $\\,\\mathcal{N}(0, I)$")
axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")
axes[0].axis("equal")
# draw 95% circle for N(0,I)
add_conf_ellipse(axes[0], np.zeros(2), I, conf=0.95, edgecolor='red', facecolor='none', lw=2, label='95%')
axes[0].legend()

# Right: after injecting two Gaussian priors (product in precision form)
axes[1].scatter(x_post[:, 0], x_post[:, 1], s=10, alpha=0.35, color='orange', label='samples')
add_conf_ellipse(axes[1], mu_post, Sigma_post, conf=0.95, edgecolor='red', facecolor='none', lw=2, label='95% ellipse')
add_conf_ellipse(axes[1], mu1, Sigma1, conf=0.95, edgecolor='green', facecolor='none', lw=2, label='95% ellipse')
add_conf_ellipse(axes[1], mu2, Sigma2, conf=0.95, edgecolor='blue', facecolor='none', lw=2, label='95% ellipse')
add_conf_ellipse(axes[1], np.zeros(2), np.eye(2), conf=0.95, edgecolor='k', facecolor='none', lw=2, label='95% ellipse')

axes[1].plot(mu_post[0], mu_post[1], 'kx', ms=10, mew=2, label='posterior mean')
axes[1].set_title("After Product: $\\mathcal{N}(\\mu_{P}, \\Sigma_{P})$")
axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2")
axes[1].axis("equal")
axes[1].legend(loc='upper left')
# limit the x and y axis
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)

plt.tight_layout()
plt.show()

# Print the resulting posterior parameters for clarity
print("Posterior mean mu_P:\n", mu_post)
print("\nPosterior covariance Sigma_P:\n", Sigma_post)
print("\nPosterior precision Lambda_P:\n", Lambda_post)
