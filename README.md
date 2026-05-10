# Perceptron & Backpropagation
> Praktikum Kecerdasan Buatan — Pertemuan 6

---

## 📌 Daftar Isi
1. [Perceptron](#1-perceptron)
   - [Definisi](#definisi)
   - [Arsitektur](#arsitektur)
   - [Fungsi Aktivasi](#fungsi-aktivasi)
   - [Algoritma Pelatihan](#algoritma-pelatihan-perceptron)
   - [Contoh Implementasi](#contoh-implementasi-perceptron)
2. [Backpropagation](#2-backpropagation)
   - [Definisi](#definisi-1)
   - [Arsitektur](#arsitektur-1)
   - [Fungsi Aktivasi](#fungsi-aktivasi-1)
   - [Algoritma Pelatihan](#algoritma-pelatihan-backpropagation)
   - [Forward Propagation](#forward-propagation)
   - [Backward Propagation](#backward-propagation)
   - [Contoh Implementasi](#contoh-implementasi-backpropagation)
3. [Perbandingan](#3-perbandingan-perceptron-vs-backpropagation)
4. [Referensi](#4-referensi)

---

## 1. Perceptron

### Definisi
**Perceptron** adalah model jaringan saraf tiruan paling sederhana yang diperkenalkan oleh Frank Rosenblatt pada tahun 1958. Perceptron merupakan model klasifikasi biner yang belajar memisahkan dua kelas data menggunakan sebuah hyperplane (garis pemisah).

Perceptron hanya terdiri dari **satu lapisan** (single-layer), sehingga hanya mampu menyelesaikan masalah yang **linearly separable** (dapat dipisahkan secara linear), seperti fungsi logika AND dan OR. Perceptron **tidak mampu** menyelesaikan masalah XOR karena XOR tidak linearly separable.

---

### Arsitektur

```
Input Layer          Output Layer
  x₁ ──w₁──┐
  x₂ ──w₂──┤──► [ Σ + bias ] ──► [ Aktivasi ] ──► ŷ
  x₃ ──w₃──┘
```

| Komponen | Deskripsi |
|----------|-----------|
| `x`      | Vektor input |
| `w`      | Vektor bobot (weight) |
| `b`      | Bias |
| `y_in`   | Weighted sum: `Σ(xᵢ·wᵢ) + b` |
| `ŷ`      | Output setelah fungsi aktivasi |

---

### Fungsi Aktivasi
Perceptron klasik menggunakan **fungsi step bipolar**:

```
         ⎧  1  jika y_in ≥ 0
ŷ  =    ⎨
         ⎩ -1  jika y_in < 0
```

Dalam implementasi Python:
```python
def predict(self, X):
    return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
```

---

### Algoritma Pelatihan Perceptron

Perceptron belajar menggunakan **Delta Rule** (Widrow-Hoff Rule):

1. **Inisialisasi** bobot `w = 0` dan bias `b = 0`
2. **Untuk setiap epoch**, lakukan:
   - Untuk setiap pasang `(xᵢ, tᵢ)`:
     1. Hitung output prediksi: `ŷ = aktivasi(w·x + b)`
     2. Hitung error: `e = t - ŷ`
     3. Update bobot: `w = w + α·e·x`
     4. Update bias: `b = b + α·e`
3. **Hitung Sum Square Error (SSE)**: `SSE = Σ eᵢ²`
4. **Berhenti** jika `SSE = 0` atau jumlah epoch maksimum tercapai

**Parameter:**
- `α` (alpha) = learning rate, mengontrol seberapa besar perubahan bobot

---

### Contoh Implementasi Perceptron

**File:** [`perceptron.py`](./perceptron.py)

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

**Hasil:** Disimpan di [`HasilPerceptron.txt`](./HasilPerceptron.txt)

---

## 2. Backpropagation

### Definisi
**Backpropagation** (Backward Propagation of Errors) adalah algoritma pelatihan untuk jaringan saraf tiruan **multi-layer** (Multi-Layer Perceptron / MLP). Algoritma ini diperkenalkan oleh Rumelhart, Hinton, dan Williams pada tahun 1986.

Backpropagation mampu menyelesaikan masalah yang **tidak linearly separable** seperti XOR, karena menggunakan **hidden layer** sebagai representasi fitur non-linear.

Proses pelatihan terdiri dari dua fase:
1. **Forward Propagation** — menghitung output dari input
2. **Backward Propagation** — menyebarkan error mundur untuk memperbarui bobot

---

### Arsitektur

```
Input Layer    Hidden Layer    Output Layer
   x₁ ────w_hidden────► h₁ ────w_output────┐
   x₂ ────w_hidden────► h₂ ────w_output────┴──► [ Aktivasi ] ──► ŷ
          + b_hidden            + b_output
```

| Lapisan | Deskripsi |
|---------|-----------|
| Input   | Data masukan (n_input = 2 neuron) |
| Hidden  | Lapisan tersembunyi (n_hidden = 2 neuron) |
| Output  | Keluaran jaringan (n_output = 1 neuron) |
| `w_hidden` | Bobot antara input dan hidden layer |
| `w_output` | Bobot antara hidden dan output layer |
| `b_hidden` | Bias hidden layer |
| `b_output` | Bias output layer |

---

### Fungsi Aktivasi

Karena data bersifat **bipolar** (bernilai -1 atau 1), digunakan **Sigmoid Bipolar (Tanh)**:

```
f(x) = tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

**Turunan** (digunakan saat backward propagation):
```
f'(x) = 1 - f(x)²  =  1 - tanh²(x)
```

Dalam implementasi Python:
```python
def bi_sigmoid(self, x):
    return np.tanh(x)

def deriv_bi_sigmoid(self, x):
    return 1 - x**2
```

> **Catatan:** Jika data bersifat **biner** (0 atau 1), gunakan **Sigmoid Biner**:
> `f(x) = 1 / (1 + e⁻ˣ)` dengan turunan `f'(x) = f(x)·(1 - f(x))`

---

### Algoritma Pelatihan Backpropagation

#### Forward Propagation

**Langkah 1 — Input ke Hidden Layer:**
```
h_in  = x · w_hidden + b_hidden
h     = tanh(h_in)
```

**Langkah 2 — Hidden ke Output Layer:**
```
y_in  = h · w_output + b_output
y     = tanh(y_in)
```

---

#### Backward Propagation

**Langkah 3 — Hitung Error Output:**
```
error = target - y
```

**Langkah 4 — Delta Output Layer:**
```
δ_output = error × (1 - y²)
```

**Langkah 5 — Error Hidden Layer:**
```
error_hidden = δ_output · w_outputᵀ
```

**Langkah 6 — Delta Hidden Layer:**
```
δ_hidden = error_hidden × (1 - h²)
```

**Langkah 7 — Update Bobot Output Layer:**
```
Δw_output = hᵀ · δ_output × α
w_output  = w_output + Δw_output

Δb_output = Σ δ_output × α
b_output  = b_output + Δb_output
```

**Langkah 8 — Update Bobot Hidden Layer:**
```
Δw_hidden = x · δ_hidden × α
w_hidden  = w_hidden + Δw_hidden

Δb_hidden = Σ δ_hidden × α
b_hidden  = b_hidden + Δb_hidden
```

**Langkah 9 — Hitung SSE & Cek Kondisi Berhenti:**
```
SSE = Σ error²
Berhenti jika SSE < target_error atau epoch maksimum tercapai
```

---

### Contoh Implementasi Backpropagation

**File:** [`Backpropagation_xor.py`](./Backpropagation_xor.py)

```python
import numpy as np
import Backpropagation as b

# Data input dan target (masalah XOR bipolar)
X = np.array([[ 1,  1],
              [ 1, -1],
              [-1,  1],
              [-1, -1]])

t = np.array([[-1],   # XOR: 1 XOR 1  = 0 → -1
              [ 1],   # XOR: 1 XOR -1 = 1 →  1
              [ 1],   # XOR: -1 XOR 1 = 1 →  1
              [-1]])  # XOR: -1 XOR -1= 0 → -1

# Pemanggilan model Backpropagation
model = b.Backpropagation(alpha=0.3, epoch=1000, target_error=0.001)
model.fit(X, t)
```

**Hasil:** Disimpan di [`hasilBackpropagation.txt`](./hasilBackpropagation.txt)

---

## 3. Perbandingan Perceptron vs Backpropagation

| Aspek | Perceptron | Backpropagation |
|-------|-----------|-----------------|
| **Jumlah Layer** | 1 (single-layer) | ≥ 3 (multi-layer) |
| **Hidden Layer** | ❌ Tidak ada | ✅ Ada |
| **Masalah** | Linearly separable (AND, OR) | Non-linear (XOR, dll.) |
| **Fungsi Aktivasi** | Step / Threshold | Sigmoid, Tanh, ReLU |
| **Update Bobot** | Delta Rule | Gradient Descent |
| **Error Propagation** | Langsung | Mundur (backward) |
| **Kompleksitas** | Rendah | Lebih tinggi |
| **Konvergensi** | Cepat (jika data separable) | Bergantung pada SSE & epoch |
| **File** | `perceptron.py` | `Backpropagation.py` |
| **Output** | `HasilPerceptron.txt` + grafik | `hasilBackpropagation.txt` + grafik |

---

## 4. Referensi

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature, 323, 533–536.
- Fausett, L. (1994). *Fundamentals of Neural Networks: Architectures, Algorithms, and Applications.* Prentice-Hall.
- Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.). Prentice-Hall.
#   H 1 D 0 2 4 1 1 1 - P r a k t i k u m K B - P e r t e m u a n 6  
 