from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from tokenizers import Tokenizer
from collections import Counter

kategoriler = ["magazin", "saglik"]

def veri_hazirlama() -> list[dict[str, str]]:
    # Etiketli veri hazırlanmalı
    veri_klasoru = Path("veri/42bin_haber/news")
    veriler = []

    for kategori in kategoriler:
        kategori_klasoru = veri_klasoru / kategori
        for dosya in kategori_klasoru.glob("*.txt"):
            with dosya.open("r", encoding="utf-8") as f:
                dosya_icerigi = f.readlines()
            # Enter karakterinden (\n) kurtulalım
            dosya_icerigi = [satir.strip() for satir in dosya_icerigi]
            dosya_icerigi = " ".join(dosya_icerigi)

            veriler.append({
                "kategori": kategori,
                "dosya_icerigi": dosya_icerigi,
            })
    return veriler

def sayisallastir(veriler: list[dict[str, str]]) -> tuple[list[list[int]], list[int]]:
    metinler = [veri["dosya_icerigi"] for veri in veriler]
    siniflar = [kategoriler.index(veri["kategori"]) for veri in veriler]

    tokenizer = Tokenizer.from_file("tokenizer.json")

    sayisal_metinler = []
    for metin in metinler:
        sayisal_metinler.append(tokenizer.encode(metin).ids)

    return sayisal_metinler, siniflar, tokenizer.get_vocab()

def hesapla_tfidf_ve_sparse(sayisal_orneklem: list[list[int]], sozluk: list[str]) -> csr_matrix:
    frekans_orneklem = [Counter(dokuman) for dokuman in sayisal_orneklem]
    N = len(sayisal_orneklem)
    M = len(sozluk)
    tdm = np.zeros((N, M))
    for i, frekanslar in enumerate(frekans_orneklem):
        avgfik = sum(frekanslar.values()) / len(frekanslar)
        katsayi = 1 / (1 + np.log10(avgfik))
        for j, fik in frekanslar.items():
            tdm[i, j] = (1 + np.log10(fik)) * katsayi
    A = tdm > 0
    df = A.sum(axis=0)
    idf = np.log10(N / df)
    tfidf = tdm * idf
    dokuman_uzunluklari = (tfidf ** 2).sum(axis=1)
    for i in range(N):
        tfidf[i, :] = tfidf[i, :] / np.sqrt(dokuman_uzunluklari[i])
    tfidf_sparse = csr_matrix(tfidf)
    return tfidf_sparse

veriler = veri_hazirlama()
sayisal_metinler, siniflar, sozluk = sayisallastir(veriler)
tfidf_sparse = hesapla_tfidf_ve_sparse(sayisal_metinler, sozluk)

