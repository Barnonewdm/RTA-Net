import numpy as np
import sys
num_samples = 11
desired_mean = np.float32(sys.argv[1])
desired_std_dev = np.float32(sys.argv[2])

samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=num_samples)

actual_mean = np.mean(samples)
actual_std = np.std(samples)
print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))

zero_mean_samples = samples - (actual_mean)

zero_mean_mean = np.mean(zero_mean_samples)
zero_mean_std = np.std(zero_mean_samples)
print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std)
scaled_mean = np.mean(scaled_samples)
scaled_std = np.std(scaled_samples)
print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

final_samples = scaled_samples + desired_mean
final_mean = np.mean(final_samples)
final_std = np.std(final_samples)
print("Final samples stats     : mean = {:.4f} stdv = {:.4f}".format(final_mean, final_std))
print(final_samples)
