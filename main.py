import argparse
import  bisect
import math
import multiprocessing
import  os
import pickle
import re
import unicodedata
import unidecode
from collections import Counter
import  json
from datetime import datetime
from json import loads
from pathlib import Path
import nltk
import numpy as np
from scipy.sparse import lil_matrix,save_npz,load_npz,csr_matrix

print("1- Veri Yükleme")

genel_baslangic = datetime.now()

baslangic = genel_baslangic

orneklem = []

#veri_klasoru =  Path(__file__).parent / "veri" / "42bin_haber" / "news" / "spor"
#for dosya_yolu in veri_klasoru.glob("*.txt"):
#    with dosya_yolu.open("r",encoding="utf-8") as dosya :
#        orneklem.append(dosya)

for dosya_adi in Path("veri/42bin_haber/news/spor").iterdir():
    metin = dosya_adi.read_text(encoding="utf-8")
    orneklem.append(metin)

print(f"Örneklem sayısı {len(orneklem)} tamamlanma süresi: {datetime.now() - baslangic}")

print("1.1- Doküman Ayıklama")

baslangic = datetime.now()

orneklem = [str(ornek) for ornek in orneklem]

print(f"Ayıklanan doküman sayısı {len(orneklem)} tamamlanma süresi: {datetime.now() - baslangic}")


print("2.1- Karakter Önişleme")

baslangic = datetime.now()

orneklem = [metin.lower() for metin in orneklem]

orneklem = [unicodedata.normalize('NFC', metin) for metin in orneklem]

secilen_kategoriler = ['Ll', 'Nd', 'Zs']

for i, ornek in enumerate(orneklem):

  kategoriler = [unicodedata.category(karakter) for karakter in ornek]

  yeni_metin = "".join([ornek[j] if kategoriler[j] in secilen_kategoriler and kategoriler[j] != 'Zs'

    else ' ' for j in range(len(ornek))])

  yeni_metin = re.sub(' +', ' ', yeni_metin)

  orneklem[i] = yeni_metin.strip()


print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("2.2- Metin Parçalama")

baslangic = datetime.now()

orneklem = [ornek.split(' ') for ornek in orneklem]

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("2.3- Metin Parçası Önişleme")

baslangic = datetime.now()

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

budayici = WordNetLemmatizer()

zamirler = set(stopwords.words('turkish'))

orneklem = [[budayici.lemmatize(parca) for parca in ornek if parca not in zamirler] for ornek in orneklem]

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("3- Sayısallaştırma")

print("3.1- Sözlük Oluşturma")

baslangic = datetime.now()

sozluk = set()

for ornek in orneklem:
    sozluk.update(ornek)

sozluk = list(sozluk)

sozluk.sort()

print(f"Sözlük boyutu {len(sozluk)} tamamlanma süresi: {datetime.now() - baslangic}")

print("3.2- Sayısallaştırma")

baslangic = datetime.now()

sayisal_orneklem = []

for ornek in orneklem:
    sayisal_ornek = [bisect.bisect_left(sozluk, kelime) for kelime in ornek]
    sayisal_orneklem.append(sayisal_ornek)

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("3.2.1- Doküman Frekansaları Hesaplanıyor")

baslangic = datetime.now()

frekans_orneklem = [Counter(dokuman) for dokuman in sayisal_orneklem]

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("3.2.1.1- Terim Frekansları Hesaplanıyor")

N = len(orneklem)
M = len(sozluk)

tdm = np.zeros((N, M))

for i, frekanslar in enumerate(frekans_orneklem):
    avgfik = sum(frekanslar.values()) / len(frekanslar)
    katsayi = 1 / (1 + np.log10(avgfik))
    for j, fik in frekanslar.items():
        tdm[i, j] = (1 + np.log10(fik)) * katsayi
print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

baslangic = datetime.now()
print("3.2.1.1- Doküman Frekansları Hesaplanıyor")

A = tdm > 0

df = A.sum(axis=0)

idf = np.log10(N / df)

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")


tfidf = tdm * idf

baslangic = datetime.now()
print("3.2.2- Doküman Vektörü Normalizasyonu")

dokuman_uzunluklari = (tfidf ** 2).sum(axis=1)

for i in range(N):
    tfidf[i, :] = tfidf[i, :] / np.sqrt(dokuman_uzunluklari[i])

print("3.3- Sparse Matrise çevirme")

tfidf_sparse = csr_matrix(tfidf)

d5 = tfidf_sparse[5, :]

d5 = tfidf_sparse.data[tfidf_sparse.indptr[5]:tfidf_sparse.indptr[6]]

print(f"Genel Tamamlanma süresi : {datetime.now() - genel_baslangic}")




















