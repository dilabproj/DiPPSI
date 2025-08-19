import numpy as np
import matplotlib.pyplot as plt

# Load data from .npy files
mse_losses = np.load('out/scientisst_ecg_test/lininterp_scientisst_ecg/mse_losses_bootstraplist.npy')
f1_scores = np.load('out/scientisst_ecg_test/lininterp_scientisst_ecg/f1_bootstraplist.npy')
precisions = np.load('out/scientisst_ecg_test/lininterp_scientisst_ecg/prec_bootstraplist.npy')
sensitivities = np.load('out/scientisst_ecg_test/lininterp_scientisst_ecg/sens_bootstraplist.npy')

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 20))  # 4 plots in a column

# MSE Losses
axs[0].plot(mse_losses, label='MSE Losses')
axs[0].set_title('MSE Losses Bootstrap Distribution')
axs[0].set_xlabel('Bootstrap Sample')
axs[0].set_ylabel('MSE Loss')
axs[0].legend()

# F1 Scores
axs[1].plot(f1_scores, label='F1 Scores', color='orange')
axs[1].set_title('F1 Scores Bootstrap Distribution')
axs[1].set_xlabel('Bootstrap Sample')
axs[1].set_ylabel('F1 Score')
axs[1].legend()

# Precisions
axs[2].plot(precisions, label='Precisions', color='green')
axs[2].set_title('Precisions Bootstrap Distribution')
axs[2].set_xlabel('Bootstrap Sample')
axs[2].set_ylabel('Precision')
axs[2].legend()

# Sensitivities
axs[3].plot(sensitivities, label='Sensitivities', color='red')
axs[3].set_title('Sensitivities Bootstrap Distribution')
axs[3].set_xlabel('Bootstrap Sample')
axs[3].set_ylabel('Sensitivity')
axs[3].legend()

# Adjust layout to make room for the plots
plt.tight_layout()

# Save the figure
plt.savefig('bootstrap_distributions.png', dpi=300)  # Save as a PNG file with high resolution