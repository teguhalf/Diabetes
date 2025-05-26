# Laporan Proyek Machine Learning - Muhammad Teguh Alfian

## Domain Proyek: Prediksi Diabetes Menggunakan CNN

Diabetes merupakan salah satu penyakit kronis yang memiliki penderita sangat banyak di seluruh dunia. Menurut World Health Organization (WHO), jumlah penderita diabetes terus meningkat setiap tahunnya, dengan estimasi mencapai ratusan juta orang. Penyakit ini tidak hanya menurunkan kualitas hidup penderitanya, tetapi juga dapat memicu komplikasi serius seperti gagal ginjal, penyakit jantung, hingga amputasi. Oleh karena itu, deteksi dini terhadap diabetes sangat penting agar penanganan medis dapat dilakukan sedini mungkin.

Di era digital dan kemajuan teknologi saat ini, pemanfaatan kecerdasan buatan (Artificial Intelligence/AI), khususnya deep learning, menjadi pendekatan yang menjanjikan dalam bidang kesehatan. Salah satu model deep learning yang telah terbukti efektif dalam pengolahan data visual dan spasial adalah Convolutional Neural Network (CNN). Meskipun CNN umumnya digunakan dalam pengenalan gambar, kemampuannya dalam mengekstraksi fitur yang kompleks juga dapat dimanfaatkan dalam prediksi penyakit, termasuk diabetes, terutama jika data telah diubah ke bentuk yang sesuai, seperti citra atau representasi spasial dari data tabular. Tujuan penelitian ini adalah untuk implementasi CNN dalam prediksi penyakit diabetes berdasarkan data yang sudah dipersiapkan.

## Business Understanding

### Problem Statements
- Jumlah penderita diabetes terus meningkat secara global setiap tahunnya, yang menimbulkan beban besar terhadap sistem kesehatan dan kualitas hidup masyarakat.
- Banyak kasus diabetes terdeteksi terlambat, sehingga meningkatkan risiko komplikasi serius seperti gagal ginjal, penyakit jantung, hingga amputasi.
- Metode konvensional dalam diagnosis dini diabetes masih memiliki keterbatasan dalam hal akurasi, kecepatan, dan skalabilitas.

### Goals
- Membangun model berbasis deep learning Convolutional Neural Network (CNN), untuk membantu proses deteksi dini diabetes secara lebih akurat dan efisien.
- Mengonversi data tabular terkait diabetes menjadi bentuk visual atau spasial yang dapat diproses oleh CNN, sehingga memungkinkan ekstraksi fitur yang lebih kompleks.
- Mengoptimalkan model CNN untuk meningkatkan performa diagnosis dalam hal akurasi, sensitivitas, dan waktu prediksi.

### Solution statements
- Menggunakan CNN Multilayer Perceptron untuk prediksi diabetes.
- Evaluasi performa model dengan accuracy, loss, dan confusion matrix

## Data Understanding
Dataset dalam penelitian ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuan dari dikumpulkannya data ini adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes, berdasarkan pengukuran diagnostik tertentu yang disertakan dalam kumpulan data. Dataset bersifat publik dengan format CSV (Comma Separated Value) yang dapat diakses melalui link berikut: [Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Diabetetes Dataset adalah sebagai berikut:
- Pregnancies              : Untuk menyatakan jumlah kehamilan
- Glucose                  : Untuk menyatakan kadar glukosa dalam darah
- BloodPressure            : Untuk menyatakan pengukuran tekanan darah
- SkinThickness	           : Untuk menyatakan ketebalan kulit
- Insulin			             : Untuk menyatakan kadar insulin dalam darah
- BMI			                 : Untuk menyatakan indeks massa tubuh
- DiabetesPedigreeFunction : Untuk menyatakan persentase diabetes
- Age			                 : Untuk menyatakan usia
- Outcome		               : Untuk menyatakan hasil akhir 1 adalah ya dan 0 adalah tidak


### EDA (Exploratory Data Analysis)
- Memahami informasi tipe data, nama kolom, jumlah baris, dan kolom dataset.<br>
![image](https://github.com/user-attachments/assets/00f11648-5fff-4089-85a4-f4d6b038679d)<br>
- Memahami deskripsi statistik dataset. <br>
![image](https://github.com/user-attachments/assets/4e17d26e-fb49-4451-af40-d1badd52644a)<br>
- Visualisasi Diagram Histogram untuk mengetahui distribusi data tiap kolom. <br>
![image](https://github.com/user-attachments/assets/83a4ebb0-be76-4704-bd93-76b9f0eaaba7)<br>
- Visualisasi Diagram Pie untuk mengetahui distribusi rentang usia (remaja, dewasa, tua).<br>
![image](https://github.com/user-attachments/assets/1a08c6b4-a95c-4f47-8f6d-76eea6e6b431)<br>
- Visualisasi Diagram Pie untuk mengetahui distribusi kategori tekanan darah (rendah, normal, tinggi).<br>
![image](https://github.com/user-attachments/assets/90d8e618-fb01-4120-871e-c46c90a18043)<br>
## Data Preparation
- Cleaning Data:
  - Mengubah nilai 0 pada beberapa kolom (‘Glucose’, ‘BloodPressure’, ‘SkinThickness’, ‘Insulin’, dan ‘BMI’) dengan nilai mean. Tujuannya yaitu merasionalkan data. Karena manusia yang masih hidup tidak mungkin memiliki nilai 0 pada kolom yang sudah disebutkan.
  - Penanganan Outlier. Metode yang digunakan adalah Interquartil. Metode ini menetapkan nilai upper bound (batas atas) untuk nilai outlier yang melebihi batas atas dan nilai lower bound (batas bawah) untuk nilai outlier yang kurang dari batas bawah. Tujuannya supaya tidak mengakibatkan bias.
  - Standardisasi menggunakan Quantil Transformer. Metode ini bertujuan mengubah distribusi variabel yang dipilih menjadi distribusi seragam. Sebelum melakukan standardisasi, dataset diubah menjadi variabel X dan variabel Y. Variabel X berisi kolom-kolom selain ‘Outcome’, sedangkan variabel Y berisi kolom ‘Outcome’ saja. Pada tahap ini, variabel X merupakan yang diterapkan standardisasi karena kolom-kolomnya tidak cukup rata distribusinya.
  - Data Splitting. Dataset dibagi menjadi tiga bagian yaitu data train, data validation, dan data test. Pada baris kode pertama, dapat dilihat bahwa data test mengambil 30% dari jumlah dataset,  sedangkan sisanya yaitu 70% akan masuk ke dalam variabel ‘temp’. Selanjutnya, pada baris kode kedua akan dibagi lagi menjadi data train dan data validation dari variabel ‘temp’. Proporsi pembagiannya adalah 30% dari 70% dialokasikan ke data validation, 70% dari 70% akan dialokasikan ke data train. Sehingga dapat disimpulkan bahwa, proporsi pembagian data yaitu 49% untuk data training, 21% untuk data validation, dan 30% untuk data testing. 

## Modeling
### Model
- Convolutional Neural Network: salah satu jenis arsitektur Deep Learning yang dapat digunakan untuk melakukan tugas prediksi. CNN dapat dibangun dengan menggunakan modul TensorFlow. Modul ini menyediakan API yang kuat dan fleksibel untuk membangun dan melatih model Deep Learning.
  - Input Layer: Menerima dan menahan data masukan yang akan diproses oleh jaringan. 
  - Dropout Layer: Teknik regularisasi yang bekerja dengan mengabaikan secara acak sejumlah neuron selama training pad setiap iterasi.
  - Hidden Layer: Melakukan sebagian besar komputasi dalam neural network.
  - Output Layer: Menyediakan tempat keluaran akhir dari neural network.
  - Optimizer: Metode yang digunakan untuk menyesuaikan weight dan bias dengan tujuan untuk mengurangi loss function.
  - Learning Rate: Nilai numerik yang mengontrol seberapa besar penyesuaian weight network sebagai respons terhadap gradien loss function.
### Flow
- Sebelum merancang model, kita perlu melakukan hyperparameter tuning. Meskipun tidak wajib, namun hyperparameter tuning membantu kita untuk menemukan parameter terbaik dengan melakukan iterasi (epoch) dengan parameter yang berbeda-beda. Dalam penelitian ini, digunakan library keras_tuner untuk membantu melakukan tuning.
![image](https://github.com/user-attachments/assets/501f5d24-ccc5-4055-a11d-76ce70871acd)<br>
![image](https://github.com/user-attachments/assets/60bc0f22-57a7-4188-8fce-4f3c5a29f07e)<br>
- Setelah mendapatkan hyperparameter dari tuning, tentunya kita akan membuat model CNN untuk memprediksi diabetes. Gunakan callbacks untuk melakukan early stopping (pengehentian awal) dan checkpoint (menyimpan model terbaik ke dalam file dengan ekstensi h5).
![image](https://github.com/user-attachments/assets/12988948-9b2f-46c5-9692-89fbbe52f592)
- Lakukan iterasi setelah membangun model CNN. Iterasi yang diterapkan pada model yaitu 20 kali.
![image](https://github.com/user-attachments/assets/b6f49d09-d030-46e4-8e39-f4e36d59a9bb)
- Setelah iterasi selesai, dapat diketahui seberapa baik performa model dengan accuracy dan loss. Tujuan dari accuracy adalah untuk mengukur kebenaran prediksi model dan loss untuk mengukur kesalahan yang dibuat oleh model.
![image](https://github.com/user-attachments/assets/981ec5f5-a6b2-43e8-96b2-3bc686a33aef)

## Evaluation
Metrik yang digunakan untuk evaluasi adalah accuracy, loss function, dan confusion matrix (precision, recall, f1-score, support)
### Penjelasan
- Accuracy: Metrik yang mengukur proporsi prediksi yang benar dari total prediksi yang dibuat oleh model.
- Loss Function: Metrik yang mengukur seberapa buruk kinerja model pada suatu data.
- Confusion Matrix: Tabel yang digunakan untuk menggambarkan kinerja model pada sekumpulan data uji yang memiliki nilai true (sebenarnya) yang diketahui. 
  - Precision: Metrik yang mengukur keakuratan dari prediksi positif model
  ![image](https://github.com/user-attachments/assets/cfd7b721-36b6-4403-96bf-6b968f89fc09)
  - Recall: Metrik yang mengukur kemampuan model untuk menemukan semua instans positif yang relevan.
  ![image](https://github.com/user-attachments/assets/9537bac9-b8f8-463f-a22c-6195ff6fbd2d)
  - F1-Score: Rata-rata harmonik dari Precision dan Recall
  ![image](https://github.com/user-attachments/assets/0255bd73-44af-4126-ae66-12c00d440c34)
  - Support: Jumlah aktual dari kemunculan setiap kelas dalam dataset yang sedang dievaluasi
### Hasil Tes
- GRAFIK HASIL ITERASI
![image](https://github.com/user-attachments/assets/5adc0c04-135d-4455-a8cd-16e9c3f6c9ad)
- Grafik Kiri: Akurasi Selama Pelatihan
  - Training Accuracy meningkat secara bertahap dari ~0.60 ke ~0.77.
  - Validation Accuracy juga meningkat, bahkan mencapai ~0.80 di epoch terakhir.
  - Model belajar dengan baik, tidak mengalami overfitting karena akurasi validasi tidak menurun meskipun akurasi training naik.
- Grafik Kanan: Loss Selama Pelatihan
  - Training Loss dan Validation Loss turun signifikan dan cukup sejajar.
  - Validation loss terus menurun hingga akhir epoch, pertanda model terus membaik.
  - Model berhasil mengurangi error baik di data pelatihan maupun validasi
  - Tidak terjadi overfitting maupun underfitting.
![image](https://github.com/user-attachments/assets/342a8ebc-913e-475e-8376-d108e1739f73)
- Kelas 0 (Negatif / Tidak Diabetes):
  - Precision: 0.82 → Dari semua prediksi "tidak diabetes", 82% benar.
  - Recall: 0.72 → Dari semua kasus nyata "tidak diabetes", 72% berhasil dikenali.
  - F1-score: 0.77 → Gabungan keseimbangan precision & recall yang cukup tinggi.
- Kelas 1 (Positif / Diabetes):
  - Precision: 0.57 → Dari semua prediksi "diabetes", hanya 57% yang benar.
  - Recall: 0.71 → Dari semua kasus nyata "diabetes", 71% berhasil terdeteksi.
  - F1-score: 0.63 → Performanya lebih rendah dibanding kelas 0, namun recall masih cukup baik.
### Kesimpulan
- Model memiliki akurasi yang cukup baik (71%), tapi masih bisa ditingkatkan.
- Model lebih bagus dalam mengenali kelas 0 (tidak diabetes) daripada kelas 1 (diabetes).
- Recall untuk kelas 1 cukup tinggi (0.71) → ini bagus untuk deteksi penyakit, karena model berhasil menangkap sebagian besar penderita diabetes.
- Precision untuk kelas 1 rendah (0.57) → artinya banyak false positive: orang yang diprediksi menderita diabetes, tapi sebenarnya tidak.
- F1-score kelas 1 = 0.63 → performa model dalam mendeteksi diabetes masih cukup, tapi perlu ditingkatkan agar lebih andal.
