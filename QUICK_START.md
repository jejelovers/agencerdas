# 🚀 Quick Start Guide - 135 Dataset

## ⚡ Pilih Versi Anda

### 🎯 RECOMMENDED: Versi OPTIMIZED
**File:** `agencerdasv1_0.py`

**Upload ke Google Colab dan run:**
```python
# Tidak perlu ubah apa-apa!
# Sudah optimized untuk 135 files
# Langsung jalankan saja
```

**Waktu training:** ~1-2 jam (Tesla T4)
**Kualitas:** Sangat baik ✅

---

### ⚡⚡⚡ Jika Masih Lambat: ULTRA-FAST
**File:** `agencerdasv1_0_ultra_fast.py`

**Upload ke Google Colab dan run:**
```python
# Versi super cepat
# Quality sedikit lebih rendah tapi jauh lebih cepat
```

**Waktu training:** ~30-45 menit (Tesla T4)
**Kualitas:** Cukup baik ⚡

---

## 📋 Langkah-Langkah

### 1. Persiapan
```bash
# Di Google Colab:
1. Upload file .py ke Colab
2. Pastikan folder audio di Google Drive: 
   /MyDrive/Agen cerdas/AudioK2/
3. Run All Cells
```

### 2. First Run (Lebih Lama)
```
✓ Install packages (1-2 menit)
✓ Mount Drive (minta izin)
✓ Load audio files (1 menit)
✓ Generate labels (5-10 menit untuk 135 files) ⏰
✓ Training (waktu tergantung versi)
✓ Save model
✓ Generate RTTM files
```

### 3. Next Runs (Lebih Cepat!)
```
✓ Install packages (1-2 menit)
✓ Mount Drive
✓ Load audio files (1 menit)
✓ Load labels from CACHE (10 detik!) ⚡⚡⚡
✓ Training (waktu sama)
✓ Done!
```

**Cache membuat run berikutnya JAUH lebih cepat!**

---

## 🔥 Tips Cepat

### Jika Out of Memory (OOM):
Edit di bagian configuration:
```python
BATCH_SIZE = 2  # Ubah dari 4 ke 2
```

### Jika Terlalu Lambat:
Gunakan ULTRA-FAST version atau test dengan subset:
```python
# Di bagian load audio, tambahkan:
audio_files = audio_files[:50]  # Test 50 files dulu
```

### Mau Training Cepat Sekarang?
```python
# Di configuration, ubah:
NUM_EPOCHS = 5  # Test dengan 5 epochs dulu
```

---

## 📊 Progress Monitoring

**Versi Optimized include progress bar:**
```
Epoch 1/15: 100%|████████| 34/34 [02:15<00:00, loss=0.4521]
Epoch 2/15: 100%|████████| 34/34 [02:10<00:00, loss=0.3845]
...
```

**Lihat:**
- Loss turun = good ✅
- Time per epoch = estimasi total
- Progress bar = tidak hang

---

## ✅ Output Files

Setelah selesai, Anda dapat:

### 1. Model
```
/content/drive/MyDrive/Agen cerdas/Model/saeend_model.pth
```
atau
```
/content/drive/MyDrive/Agen cerdas/Model/ultrafast_model.pth
```

### 2. RTTM Files (hasil diarizasi)
```
/content/drive/MyDrive/Agen cerdas/RTTM Output/*.rttm
```
Satu file .rttm untuk setiap audio file

### 3. Cache (untuk run berikutnya)
```
/content/drive/MyDrive/Agen cerdas/Cache/labels_cache_*.pkl
```

---

## 🎯 Ringkasan

| Kebutuhan | Gunakan | Waktu |
|-----------|---------|-------|
| Hasil terbaik, waktu OK | **OPTIMIZED** | 1-2 jam |
| Butuh cepat sekarang | **ULTRA-FAST** | 30-45 min |
| Test coba-coba | OPTIMIZED + subset 20 files | 10 min |
| Production final | OPTIMIZED full | 1-2 jam |

---

## 🆘 Quick Troubleshooting

### Tidak mulai-mulai?
- **Tunggu** - sedang generate labels (5-10 menit pertama)
- Lihat output, ada progress

### CUDA OOM?
```python
BATCH_SIZE = 2
```

### Terlalu lambat?
- Gunakan ULTRA-FAST version
- Atau subset data

### Loss tidak turun?
- Normal di awal epoch
- Tunggu sampai epoch 3-5
- Loss harus turun gradually

---

## 💪 Siap Mulai?

1. **Upload `agencerdasv1_0.py` ke Colab**
2. **Run All** (Runtime > Run all)
3. **Tunggu...**
4. **Done!** ✅

Jika ada masalah, baca `OPTIMIZATION_GUIDE.md` untuk detail lengkap.

**Selamat mencoba! 🎉**
