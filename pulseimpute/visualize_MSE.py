import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming mse_data is a dictionary containing arrays of MSE values for each method
mse_data = {
    'Mean': np.load('out/scientisst_ecg_test/mean_scientisst_ecg/mse_losses_bootstraplist.npy'),
    'Lin Interp': np.load('out/scientisst_ecg_test/lininterp_scientisst_ecg/mse_losses_bootstraplist.npy'),
    'FFT': np.load('out/scientisst_ecg_test/fft_scientisst_ecg/mse_losses_bootstraplist.npy')
    # ... add other methods similarly
}

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Draw the plots
for label, mse_vals in mse_data.items():
    sns.kdeplot(mse_vals, label=label, bw_adjust=0.5)  # bw_adjust smoothes or sharpens the curve

# Final plot adjustments
plt.title('MSE Distribution for Various Imputation Methods')
plt.xlabel('MSE for a Given Waveform')
plt.ylabel('Frequency')
plt.legend(title='Method')
plt.tight_layout()

# Save the plot
plt.savefig('mse_distribution_plot.png')
