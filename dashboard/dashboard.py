import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
day_df = pd.read_csv("data/day.csv")
hour_df = pd.read_csv("data/hour.csv")

# Set up Streamlit page
st.title("Dashboard Analisis Penyewaan Sepeda :bike:")
st.write("Analisis ini menggunakan dataset penyewaan sepeda harian dan per jam untuk mengeksplorasi faktor-faktor yang mempengaruhi penyewaan sepeda.")

# Section 1: Tinjauan Dataset
st.header("1. Tinjauan Dataset")
st.subheader("Contoh data dari dataset day.csv")
st.write(day_df.head())

st.subheader("Contoh data dari dataset hour.csv")
st.write(hour_df.head())

# Section 2: Wawasan Utama
st.header("2. Wawasan Utama")
st.write("""
- **Tidak ada missing value** pada kedua dataset (day.csv dan hour.csv).
- **Tidak ada data duplikat** yang ditemukan di kedua dataset.
- Terdapat **outlier** pada dataset `hour.csv`, namun **tidak ada outlier** pada dataset `day.csv`.
""")

# Menghapus kolom yang tidak relevan untuk perhitungan korelasi
day_df_num = day_df.drop(columns=['dteday'])

# Hitung matriks korelasi
correlation = day_df_num.corr()

# Visualisasi heatmap untuk korelasi
st.header("3. Korelasi Antar Variabel (day.csv)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Section 4: Penyewaan Sepeda Berdasarkan Suhu
st.header("4. Penyewaan Sepeda Berdasarkan Suhu")

# Membuat kategori suhu
bins = [0, 10, 20, 30, 40]
labels = ['0-10°C', '10-20°C', '20-30°C', '30-40°C']
day_df['temp_category'] = pd.cut(day_df['temp'] * 40, bins=bins, labels=labels, include_lowest=True)

# Hitung rata-rata penyewaan sepeda untuk setiap kategori suhu
average_rentals = day_df.groupby('temp_category')['cnt'].mean()

# Buat bar plot untuk penyewaan berdasarkan suhu
fig, ax = plt.subplots(figsize=(10, 6))
average_rentals.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Rata-Rata Penyewaan Sepeda Berdasarkan Rentang Suhu')
ax.set_xlabel('Rentang Suhu')
ax.set_ylabel('Rata-Rata Penyewaan Sepeda')
st.pyplot(fig)

# Section 5: Penyewaan Sepeda Berdasarkan Jam
st.header("5. Penyewaan Sepeda Berdasarkan Jam (hour.csv)")

# Mengelompokkan data berdasarkan jam dan menghitung total penyewaan
hourly_counts = hour_df.groupby('hr')['cnt'].sum()

# Bar plot untuk penyewaan sepeda per jam
fig, ax = plt.subplots(figsize=(10, 6))
hourly_counts.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Total Penyewaan Sepeda Berdasarkan Jam')
ax.set_xlabel('Jam dalam Sehari')
ax.set_ylabel('Total Penyewaan Sepeda')
st.pyplot(fig)

# Section 6: Clustering Manual (Jam Sibuk vs Jam Tidak Sibuk)
st.header("6. Clustering Manual: Jam Sibuk vs Jam Tidak Sibuk")

# Membuat clustering sederhana berdasarkan threshold penyewaan
threshold = 20000
hourly_data = hour_df.groupby('hr')['cnt'].sum().reset_index()
hourly_data['cluster'] = hourly_data['cnt'].apply(lambda x: 'Jam Sibuk' if x > threshold else 'Jam Tidak Sibuk')

# Bar plot untuk clustering manual
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(hourly_data['hr'], hourly_data['cnt'], color=hourly_data['cluster'].map({'Jam Sibuk': 'red', 'Jam Tidak Sibuk': 'blue'}))
ax.set_title('Jam Sibuk vs Jam Tidak Sibuk Berdasarkan Penyewaan Sepeda')
ax.set_xlabel('Jam')
ax.set_ylabel('Jumlah Penyewaan Sepeda')
st.pyplot(fig)

# Section 7: Distribusi Penyewaan Sepeda Berdasarkan Cluster
st.header("7. Distribusi Penyewaan Sepeda Berdasarkan Cluster")

# Boxplot untuk distribusi penyewaan sepeda per cluster
hour_df['cluster'] = hour_df['hr'].map(dict(zip(hourly_data['hr'], hourly_data['cluster'])))
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='cluster', y='cnt', data=hour_df, ax=ax)
ax.set_title('Distribusi Penyewaan Sepeda Berdasarkan Cluster (Jam Sibuk vs Tidak Sibuk)')
st.pyplot(fig)

# Optional: Insight Conclusion Section
st.header("8. Kesimpulan")
st.write("""
1. Analisis menunjukkan bahwa beberapa faktor, seperti suhu, kelembapan, dan waktu, memiliki pengaruh signifikan terhadap jumlah penyewaan sepeda. Data menunjukkan bahwa pada hari yang cerah dengan suhu yang lebih tinggi, jumlah penyewaan cenderung meningkat, sementara cuaca yang buruk dan suhu yang lebih rendah cenderung mengurangi jumlah penyewaan.
2. Terdapat hubungan positif yang signifikan antara suhu dan jumlah penyewaan sepeda. Saat suhu meningkat, jumlah penyewaan sepeda juga cenderung meningkat. Hal ini menunjukkan bahwa pengguna lebih memilih untuk menyewa sepeda saat cuaca hangat, yang mungkin berkaitan dengan kenyamanan dan kenyamanan dalam berkendara.
3. Analisis menunjukkan bahwa jam penyewaan sepeda paling tinggi terjadi antara pukul 17:00 hingga 19:00, saat banyak orang pulang kerja atau sekolah. Sebaliknya, jumlah penyewaan paling rendah tercatat pada pukul 23:00 hingga 05:00, ketika sebagian besar pengguna tidak menyewa sepeda.
4. Jam sibuk untuk penyewaan sepeda terjadi pada sore hari, khususnya antara pukul 17:00 dan 18:00, sedangkan jam tidak sibuk atau sepi terjadi pada pagi hari sekitar pukul 06:00 hingga 08:00 dan malam hari setelah pukul 20:00.
""")

st.caption('Copyright © Alifia Luthfi 2024')