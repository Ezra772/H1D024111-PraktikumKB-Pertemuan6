# Import library
import numpy as np
import matplotlib.pyplot as plt
# Buat kelas Perceptron
class Perceptron:
    # Simpan learning rate dan max epoch dalam konstruktor
    def __init__(self, alpha=0.1, epoch=10):
        self.alpha = alpha
        self.epoch = epoch

    # Fungsi menghitung nilai y_in atau net
    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    # Fungsi menerapkan fungsi aktivasi bipolar
    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    # Fungsi membuat simulasi garis pemisah data
    def plot_decision_boundary(self, X, t, epoch):
        # Membuat titik data input
        plt.scatter(X[:, 0], X[:, 1], c=t.ravel(), marker='o',
        edgecolors='k', cmap=plt.cm.RdYlBu)
        
        # Menentukan limit tampilan bidang grafik
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Membuat garis pemisah
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(self.w_[0] + self.w_[1] * x_vals) / self.w_[2]
        
        plt.plot(x_vals, y_vals, 'b-', label=f'Decision boundary (Epoch{epoch+1})')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Decision Boundary Pada Epoch {epoch+1}")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()

    # Fungsi utama Perceptron
    def fit(self, X, t):
        # Inisialisasi bobot dan bias awal = 0
        self.w_ = np.zeros(1 + X.shape[1])
        # Menyimpan hasil pada HasilPerceptron.txt
        with open("HasilPerceptron.txt", "w") as f:
            f.write("Masalah OR dengan Perceptron\n")
            f.write("----------------------------\n")
            f.write(f"Input :\n{X}\n")
            f.write(f"Target:\n{t}\n")
            f.write(f"Bobot awal : {self.w_[1:]}\n")
            f.write(f"Bias awal : {self.w_[0]}\n")
            f.write(f"Learning rate : {self.alpha}\n")
            f.write(f"Max Epoch : {self.epoch}\n")

            # Iterasi Perceptron(Epoch)
            for epoch in range(self.epoch):
                f.write(f"\nEpoch {epoch + 1}/{self.epoch}\n")
                f.write("----------\n")
                errors = []

                # Iterasi setiap pasang matriks input dengan targetnya
                for xi, target in zip(X, t):
                    # Periksa input dengan model Perceptron
                    y_pred = self.predict(xi)

                    # Hitung error untuk sampel ini
                    err = (target - y_pred).item()
                    errors.append(err)

                    # Modifikasi bobot dengan Delta Rule
                    update = self.alpha * err

                    # Modifikasi bobot w
                    self.w_[1:] += update * xi

                    # Modifikasi bias b
                    self.w_[0] += update

                    # Menuliskan hasil tiap iterasi input pada satu epoch
                    f.write(f"Input: {xi}, Target: {target}, Predict: {y_pred}, Error: {err}, Bobot: {self.w_[1:]}, Bias: {self.w_[0]}\n")
                    
                # Simulasikan garis pemisah model Perceptron (setiap akhir epoch)
                self.plot_decision_boundary(X, t, epoch)

                # Menuliskan penjumlahan kuadrat error setiap input
                sse = sum(e ** 2 for e in errors)
                f.write(f"Sum Square Error(SSE): {sse}\n")

                # Periksa kondisi berhenti
                if sse == 0 or epoch + 1 == self.epoch:
                    f.write("--------------------------------------------------------------------------------\n")
                    f.write(f"Pelatihan berhenti pada epoch ke-{epoch + 1} karena ")
                    f.write("Sum Square Error(SSE) mencapai target.\n" if sse == 0 else "max epoch tercapai.\n")
                    # Menuliskan bobot terakhir
                    f.write(f"\nBobot akhir :{self.w_[1:]}\n")
                    f.write(f"Bias akhir :{self.w_[0]}")
                    break
