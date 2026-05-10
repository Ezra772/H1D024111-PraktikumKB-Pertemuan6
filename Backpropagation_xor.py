# Import library
import numpy as np
import Backpropagation as b

# Inisialisasi input dan target
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
t = np.array([[-1], [1], [1], [-1]])

# Pemanggilan model Backpropagation
model = b.Backpropagation(alpha=0.3, epoch=1000, target_error=0.001)
model.fit(X, t)
