# Import library
import numpy as np
import perceptron as p

# Inisialisasi input dan target
X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
t = np.array([[1], [1], [1], [-1]])
# Pemanggilan model Perceptron
model = p.Perceptron(alpha=0.1, epoch=10)
# Pemanggilan fungsi fit dan memulai pelatihan
model.fit(X, t)