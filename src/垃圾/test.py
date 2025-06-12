import numpy as np
a = np.ones((2, 2, 2))
print(np.fft.irfftn(np.fft.rfftn(a)))
