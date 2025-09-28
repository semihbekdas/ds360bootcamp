
# DVC ile Veri Versiyonlama

Bu proje DVC'nin temel kullanımını öğretir.
dvc: yaml dosyaıyla tüm kodları çalıştırırsın ve sonra tekrar dvc repro dersen sadece değişen kısımlar çalışır eğer bir yer değişmediyse tekrar baştan modeli eğitmez
sadece değişiklik yaptığın yere bakar

## Adımlar

1. **Veri oluştur:**
   ```bash
   python create_data.py
   ```

2. **DVC ile takip et:**
   ```bash
   dvc add data.csv
   ```

3. **Git'e commit et:**
   ```bash
   git add data.csv.dvc .gitignore create_data.py README.md
   git commit -m "İlk veri versiyonu"
   ```

4. **Veriyi değiştir ve yeni versiyon oluştur:**
   ```bash
   # create_data.py dosyasını düzenle
   python create_data.py
   dvc add data.csv
   git add data.csv.dvc
   git commit -m "Veri güncellendi - v2"
   ```

## DVC Komutları

- `dvc add dosya.csv` - Dosyayı DVC ile takip et
- `dvc status` - Değişiklikleri kontrol et
- `dvc push` - Remote storage'a yükle
- `dvc pull` - Remote storage'dan indir

dvc get https://github.com... dvc veri kümesi almak veri  kaydı lamka internetten
 dvc add data/data.xml
 git add data/.gitinore data/data.xml.dvc
 git commit -m "dvc added"

 dvc remote ile datamızı dışarı export edebil,rz
 driveile
 dvc remote add  -d storage gdrive://link
 git commit .dvc/config -m"configru remote storage"
dvc push

data.xml.dvc githuba gider


dvc pull ile çekebilirz datayı /getle farkı nedir öğren


dvc yaml nasıl oluşturulur


## 1️⃣ DVC ile Veri Versiyonlama

### Ne İşe Yarar?
- Büyük veri dosyalarını Git'te takip etmek yerine DVC ile takip ederiz
- Veri değişikliklerini versiyon olarak saklarız
- Takım çalışmasında veri senkronizasyonu sağlarız

### Temel Komutlar
```bash
cd 01-dvc-versioning

# DVC başlat
dvc init

# Veri dosyası oluştur
python create_data.py

# DVC ile takip et
dvc add data.csv

# Git'e commit et
git add data.csv.dvc .gitignore
git commit -m "İlk veri versiyonu"
```

### Önemli Dosyalar
- `data.csv.dvc` - DVC metadata dosyası (Git'te saklanır)
- `data.csv` - Gerçek veri (DVC tarafından takip edilir)
- `.dvcignore` - DVC'nin ignore ettiği dosyalar

### Pratik Örnek
```python
# create_data.py
import pandas as pd

data = {
    'isim': ['Ali', 'Ayşe', 'Mehmet', 'Fatma'],
    'yas': [25, 30, 35, 28],
    'maas': [5000, 6000, 7000, 5500]
}

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)
```
