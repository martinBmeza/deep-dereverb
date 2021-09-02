import numpy as np
import matplotlib.pyplot as plt

sdr_reverb = np.load('SDR_reverb.npy')
sdr_dereverb = np.load('SDR_dereverb.npy')

plt.subplot(1,3,1)
plt.title('SDR')
plt.hist(sdr_dereverb, bins=10,label = 'dereverb', alpha = 0.8)
plt.hist(sdr_reverb, bins=10, label = 'reverb', alpha = 0.8)
plt.legend()

srmr_reverb = np.load('SRMR_reverb.npy')
srmr_dereverb = np.load('SRMR_dereverb.npy')

plt.subplot(1,3,2)
plt.title('SRMR')
plt.hist(srmr_dereverb, bins=10,label = 'dereverb', alpha = 0.8)
plt.hist(srmr_reverb, bins=10, label = 'reverb', alpha = 0.8)
plt.legend()

estoi_reverb = np.load('ESTOI_reverb.npy')
estoi_dereverb = np.load('ESTOI_dereverb.npy')

plt.subplot(1,3,3)
plt.title('ESTOI')
plt.hist(estoi_dereverb, bins=10,label = 'dereverb', alpha = 0.8)
plt.hist(estoi_reverb, bins=10, label = 'reverb', alpha = 0.8)
plt.legend()

print('SDR Reverb: {:.2f}'.format(np.median(sdr_reverb)))
print('SDR Dereverb: {:.2f}'.format(np.median(sdr_dereverb)))
print('SRMR Reverb: {:.2f}'.format(np.median(srmr_reverb)))
print('SRMR Dereverb: {:.2f}'.format(np.median(srmr_dereverb)))
print('ESTOI Reverb: {:.2f}'.format(np.median(estoi_reverb)))
print('ESTOI Dereverb: {:.2f}'.format(np.median(estoi_dereverb)))


plt.show()
