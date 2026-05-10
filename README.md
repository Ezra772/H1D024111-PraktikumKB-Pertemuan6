# Perceptron & Backpropagation

> Praktikum Kecerdasan Buatan — Pertemuan 6

---

## Daftar Isi

1. [Perceptron](#1-perceptron)
2. [Backpropagation](#2-backpropagation)
3. [Perbandingan](#3-perbandingan-perceptron-vs-backpropagation)
4. [Referensi](#4-referensi)

---

## 1. Perceptron

### Definisi

**Perceptron** adalah model jaringan saraf tiruan paling sederhana yang diperkenalkan oleh Frank Rosenblatt pada tahun 1958. Perceptron merupakan model klasifikasi biner yang belajar memisahkan dua kelas data menggunakan sebuah hyperplane (garis pemisah).

Perceptron hanya terdiri dari **satu lapisan** (single-layer), sehingga hanya mampu menyelesaikan masalah yang **linearly separable** seperti AND dan OR. Perceptron **tidak mampu** menyelesaikan masalah XOR karena XOR tidak linearly separable.

---

### Arsitektur

```
Input Layer              Output Layer

  x1 --- w1 -+
  x2 --- w2 -+---[ Net = sum(xi*wi) + b ]---[ Aktivasi ]---> y_hat
  x3 --- w3 -+
```

| Komponen | Deskripsi |
|----------|-----------|
| `x` | Vektor input |
| `w` | Vektor bobot (weight) |
| `b` | Bias |
| `y_in` | Weighted sum: `sum(xi * wi) + b` |
| `y_hat` | Output setelah fungsi aktivasi |

---

### Fungsi Aktivasi

Perceptron klasik menggunakan **fungsi step bipolar**:

```
         +1   jika y_in >= 0
y_hat =  
         -1   jika y_in < 0
```

Implementasi Python:

```python
def predict(self, X):
    return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
```

---

### Algoritma Pelatihan Perceptron

Perceptron belajar menggunakan **Delta Rule** (Widrow-Hoff Rule):

1. **Inisialisasi** bobot `w = 0` dan bias `b = 0`
2. **Untuk setiap epoch**, ulangi:
   - Untuk setiap pasang `(xi, ti)`:
     1. Hitung prediksi: `y_hat = aktivasi(w . x + b)`
     2. Hitung error: `e = t - y_hat`
     3. Update bobot: `w = w + alpha * e * x`
     4. Update bias: `b = b + alpha * e`
3. Hitung **Sum Square Error (SSE)**: `SSE = sum(e^2)`
4. **Berhenti** jika `SSE = 0` atau epoch maksimum tercapai

> **alpha** = learning rate, mengontrol seberapa besar perubahan bobot per iterasi.

---

### Contoh Implementasi Perceptron

**File:** [perceptron_or.py](./perceptron_or.py)

```python
import numpy as np
import perceptron as p

# Data input dan target (masalah OR bipolar)
X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
t = np.array([[1], [1], [1], [-1]])

# Inisialisasi dan pelatihan model
model = p.Perceptron(alpha=0.1, epoch=10)
model.fit(X, t)
```

Hasil pelatihan disimpan di: `HasilPerceptron.txt`

---

## 2. Backpropagation

### Definisi

**Backpropagation** (Backward Propagation of Errors) adalah algoritma pelatihan untuk jaringan saraf tiruan **multi-layer** (Multi-Layer Perceptron / MLP). Diperkenalkan oleh Rumelhart, Hinton, dan Williams pada tahun 1986.

Backpropagation mampu menyelesaikan masalah yang **tidak linearly separable** seperti XOR, karena menggunakan **hidden layer** sebagai representasi fitur non-linear.

Proses pelatihan terdiri dari dua fase:
1. **Forward Propagation** — menghitung output dari input
2. **Backward Propagation** — menyebarkan error mundur untuk memperbarui bobot

---

### Arsitektur

```
Input Layer      Hidden Layer     Output Layer

  x1 --+                          
       |--- w_hidden --> h1 --+
  x2 --+                     +--- w_output --> y_hat
                         h2 --+
              + b_hidden            + b_output
```

| Lapisan | Neuron | Keterangan |
|---------|--------|------------|
| Input | 2 | `n_input = 2` |
| Hidden | 2 | `n_hidden = 2` |
| Output | 1 | `n_output = 1` |

| Variabel | Ukuran | Deskripsi |
|----------|--------|-----------|
| `w_hidden` | (2 x 2) | Bobot input ke hidden layer |
| `b_hidden` | (1 x 2) | Bias hidden layer |
| `w_output` | (2 x 1) | Bobot hidden ke output layer |
| `b_output` | (1 x 1) | Bias output layer |

---

### Fungsi Aktivasi

Karena data bersifat **bipolar** (nilai -1 atau 1), digunakan **Sigmoid Bipolar (Tanh)**:

```
f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

**Turunannya** (digunakan saat backward propagation):

```
f'(x) = 1 - f(x)^2
```

Implementasi Python:

```python
def bi_sigmoid(self, x):
    return np.tanh(x)

def deriv_bi_sigmoid(self, x):
    return 1 - x**2
```

> **Catatan:** Untuk data **biner** (nilai 0 atau 1), gunakan Sigmoid Biner:
> `f(x) = 1 / (1 + e^-x)` dengan turunan `f'(x) = f(x) * (1 - f(x))`

---

### Algoritma Pelatihan Backpropagation

#### Forward Propagation

**Langkah 1 — Input ke Hidden Layer:**

```
h_in = x . w_hidden + b_hidden
h    = tanh(h_in)
```

**Langkah 2 — Hidden ke Output Layer:**

```
y_in = h . w_output + b_output
y    = tanh(y_in)
```

#### Backward Propagation

**Langkah 3 — Hitung Error Output:**

```
error = target - y
```

**Langkah 4 — Delta Output Layer:**

```
delta_output = error * (1 - y^2)
```

**Langkah 5 — Error Hidden Layer:**

```
error_hidden = delta_output . w_output^T
```

**Langkah 6 — Delta Hidden Layer:**

```
delta_hidden = error_hidden * (1 - h^2)
```

**Langkah 7 — Update Bobot Output Layer:**

```
w_output = w_output + (h^T . delta_output) * alpha
b_output = b_output + sum(delta_output) * alpha
```

**Langkah 8 — Update Bobot Hidden Layer:**

```
w_hidden = w_hidden + (x . delta_hidden) * alpha
b_hidden = b_hidden + sum(delta_hidden) * alpha
```

**Langkah 9 — Cek Kondisi Berhenti:**

```
SSE = sum(error^2)
Berhenti jika SSE < target_error atau epoch maksimum tercapai
```

---

### Contoh Implementasi Backpropagation

**File:** [Backpropagation_xor.py](./Backpropagation_xor.py)

```python
import numpy as np
import Backpropagation as b

# Data input dan target (masalah XOR bipolar)
X = np.array([[ 1,  1],
              [ 1, -1],
              [-1,  1],
              [-1, -1]])

t = np.array([[-1],   # 1  XOR  1  = 0 -> -1
              [ 1],   # 1  XOR -1  = 1 ->  1
              [ 1],   # -1 XOR  1  = 1 ->  1
              [-1]])  # -1 XOR -1  = 0 -> -1

model = b.Backpropagation(alpha=0.3, epoch=1000, target_error=0.001)
model.fit(X, t)
```

Hasil pelatihan disimpan di: `hasilBackpropagation.txt`

---

## 3. Perbandingan Perceptron vs Backpropagation

| Aspek | Perceptron | Backpropagation |
|-------|-----------|-----------------|
| **Jumlah Layer** | 1 (single-layer) | 3+ (multi-layer) |
| **Hidden Layer** | Tidak ada | Ada |
| **Masalah** | Linearly separable (AND, OR) | Non-linear (XOR, dll.) |
| **Fungsi Aktivasi** | Step / Threshold | Tanh / Sigmoid / ReLU |
| **Update Bobot** | Delta Rule | Gradient Descent |
| **Propagasi Error** | Langsung ke output | Mundur (backward) |
| **Kompleksitas** | Rendah | Lebih tinggi |
| **File Implementasi** | `perceptron.py` | `Backpropagation.py` |
| **File Output** | `HasilPerceptron.txt` | `hasilBackpropagation.txt` |

---

## 4. Referensi

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature, 323, 533–536.
- Fausett, L. (1994). *Fundamentals of Neural Networks: Architectures, Algorithms, and Applications.* Prentice-Hall.
- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.). Prentice-Hall.