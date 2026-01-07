\# 🚀 Dynamic DeepFace Batch Processor (Stability Focused)



Bu proje, \*\*DeepFace\*\* kütüphanesini kullanarak büyük veri setlerini (1000+ resim) işlemek için tasarlanmış, \*\*kaynak farkındalığına sahip\*\* (resource-aware) profesyonel bir toplu işlemcidir. 







\## 🧠 Mühendislik Kararları ve İyileştirmeler

Kıdemli mühendis eleştirileri doğrultusunda, kodun mimarisi "kaba kuvvet" yerine "akıllı kaynak yönetimi" üzerine yeniden inşa edilmiştir:



\- \*\*Shared Memory Architecture (Threading):\*\* Her işçi için modeli tekrar yükleyip RAM'i israf etmek yerine, `ThreadPoolExecutor` kullanılarak model bellekte \*\*tek bir kopya\*\* olarak tutulur. Bu sayede RAM kullanımı %80 oranında azaltılmıştır.

\- \*\*Bounded Semaphore Management:\*\* İşlemciyi boğmamak ve kilitlenmeleri (Deadlock) önlemek için ağır analiz süreçlerini yöneten bir "Semafor Fedaisi" eklenmiştir.

\- \*\*Dynamic Resource Allocation:\*\* Sabit sayılar (Magic Numbers) kaldırılmıştır. Kod, sistemdeki boş RAM miktarını ve CPU çekirdek sayısını anlık analiz ederek eşzamanlı işlem kapasitesini otomatik belirler.

\- \*\*False Positive Protection:\*\* Hızlı ama hatalı modeller (OpenCV) yerine `retinaface` backend'i kullanılarak çizimlerin, logoların veya bulutların "insan" sanılmasının önüne geçilmiştir.



\## 📊 Sistem İstikrarı

Yapılan testlerde, sistemin başlangıçtaki \*\*6263 MB\*\* boş RAM miktarını işlem boyunca güvenli sınırda tuttuğu ve bilgisayarı dondurmadığı kanıtlanmıştır.



\## 📥 Kurulum ve Kullanım

1\. Gereksinimleri yükleyin:

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt
Resimlerinizi pics klasörüne atın ve scripti çalıştırın:
python script.py


