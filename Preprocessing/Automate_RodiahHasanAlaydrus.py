import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def auto_preprocessing(file_path, target_column=None, save_path=None):

    # 1. Memuat data
    df = pd.read_csv(file_path)

    # 2. Menghapus duplikat dan missing value
    df = df.drop_duplicates()
    df = df.dropna()

    # 3. Memisahkan target (jika ada)
    if target_column:
        y = df[target_column]
        df = df.drop(columns=[target_column])
    else:
        y = None

    # 4. Identifikasi kolom numerik dan kategorikal
    kolom_numerik = df.select_dtypes(include=np.number).columns
    kolom_kategori = df.select_dtypes(include='object').columns

    # 5. Encoding data kategorikal
    for col in kolom_kategori:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 6. Scaling data numerik
    scaler = StandardScaler()
    df[kolom_numerik] = scaler.fit_transform(df[kolom_numerik])

    # 7. Penanganan outlier dengan metode IQR
    for col in kolom_numerik:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        df[col] = df[col].clip(batas_bawah, batas_atas)

    # 8. Menggabungkan kembali kolom target
    if target_column:
        df[target_column] = y.values

    # 9. Menyimpan hasil preprocessing
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"File preprocessing berhasil disimpan di: {save_path}")

auto_preprocessing(
    file_path="insurance_raw.csv",
    target_column=None,
    save_path="Preprocessing/insurance_preprocessed.csv"
)
