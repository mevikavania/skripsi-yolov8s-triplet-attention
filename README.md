# Deteksi Tumor Otak pada Citra MRI dengan YOLOv8s dan Triplet Attention

Saya membuat aplikasi Streamlit ini sebagai bagian dari skripsi saya di Program Studi Teknik Informatika, Universitas Sriwijaya. Di aplikasi ini, saya membandingkan hasil deteksi YOLOv8s baseline dan YOLOv8s dengan Triplet Attention pada citra MRI yang sama.

**Mevika Vania · Universitas Sriwijaya · 2026**

## Dataset dan metode

Dalam penelitian ini, saya menggunakan 6.000 citra MRI dari dataset BRISC2025 dengan pembagian data pelatihan, validasi, dan pengujian sebesar 80:10:10. Model mendeteksi tiga kelas tumor, yaitu glioma, meningioma, dan pituitary. Citra `no_tumor` digunakan sebagai sampel negatif.

Saya menambahkan Triplet Attention melalui modul `C2fTriplet` pada tiga tingkat fitur backbone YOLOv8s, yaitu P3/8, P4/16, dan P5/32. Modul ini memproses hubungan antardimensi *channel–width*, *channel–height*, dan *height–width*.

## Hasil pengujian

| Metrik | YOLOv8s baseline | YOLOv8s + Triplet Attention |
| --- | ---: | ---: |
| Precision | 0,895490 | 0,914876 |
| Recall | 0,883998 | 0,911173 |
| F1-score | 0,889707 | 0,913020 |
| mAP@0.5 | 0,927308 | 0,936265 |
| mAP@0.5:0.95 | 0,574464 | 0,629379 |

Pada data uji BRISC2025, model dengan Triplet Attention memperoleh nilai lebih tinggi pada kelima metrik. Peningkatan terbesar ada pada mAP@0.5:0.95, yaitu sekitar 5,49 poin persentase.

## Menjalankan aplikasi

Dari folder repo, jalankan perintah berikut di PowerShell dengan Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka `http://localhost:8501`, lalu unggah satu citra MRI berformat JPG, JPEG, atau PNG. Aplikasi menampilkan *bounding box* kedua model secara berdampingan, serta kelas dengan *confidence score* tertinggi dan waktu inferensi masing-masing model. Aplikasi memakai ambang *confidence* 0,25.

Saya membuat aplikasi ini untuk mendemonstrasikan hasil penelitian. Hasil prediksinya tidak digunakan untuk diagnosis klinis.

## Struktur repo

| File | Isi |
| --- | --- |
| `app.py` | Antarmuka Streamlit dan alur inferensi kedua model |
| `triplet_attention.py` | Implementasi Triplet Attention dan `C2fTriplet` |
| `baseline_best.pt` | Bobot YOLOv8s baseline |
| `triplet_best.pt` | Bobot YOLOv8s dengan Triplet Attention |
| `requirements.txt` | Dependensi untuk menjalankan aplikasi |
