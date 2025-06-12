# Türkçe PDF Belgeleri Üzerinde Gelişmiş Soru-Cevap Sistemi (RAG)

Bu proje, PDF belgelerindeki bilgilere dayanarak Türkçe soruları yanıtlamak üzere tasarlanmış bir Retrieval Augmented Generation (RAG) sistemidir. Sistem, metinleri anlamlı parçalara ayırır, bu parçaları vektörleştirerek aranabilir bir indeks oluşturur, kullanıcı sorusuna en uygun metin parçalarını bulur ve bu bağlamı kullanarak Büyük Dil Modeli (LLM) ile cevap üretir. Ayrıca, üretilen cevapların kalitesini çeşitli metriklerle değerlendirir.

## 🚀 Temel Özellikler

* **PDF İşleme:** PDF dosyalarından metin içeriğini çıkarma.
* **Gelişmiş Metin Ön İşleme:** Türkçe için özel tokenizasyon ve stopword (etkisiz kelime) temizliği.
* **Esnek Metin Parçalama (Chunking):** Hem yinelemeli (recursive) hem de anlamsal (semantic) parçalama seçenekleri.
* **Vektör Veritabanı:** FAISS kullanarak verimli benzerlik araması için metin parçalarının (chunk) vektör indeksini oluşturma.
* **Model Seçenekleri:**
    * **LLM:** `mistralai/Mistral-7B-Instruct-v0.3` (4-bit quantization ile GPU'da optimize edilmiş kullanım).
    * **Embedding Modeli:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
* **RAG Mimarisi:** Soruya en uygun bağlamı bularak LLM'in daha doğru ve ilgili cevaplar üretmesini sağlama.
* **Kapsamlı Değerlendirme:** Üretilen cevapların kalitesini ROUGE, BLEU, METEOR, BERTScore, Kosinüs Benzerliği ve BLEURT gibi metriklerle değerlendirme.
* **Önbellekleme (Caching):** İşlenmiş metin parçalarını ve FAISS indekslerini diske kaydederek tekrar eden işlemlerde zaman kazanımı.
* **Yapılandırılabilirlik:** Parçalama, arama ve LLM cevap üretimi için birçok parametrenin ayarlanabilir olması.
* **Google Drive Entegrasyonu:** Verilerin ve önbelleğin Google Drive üzerinde saklanması.
* **İnteraktif Kullanım:** Kullanıcının PDF seçmesine ve art arda sorular sormasına olanak tanıyan arayüz.
* **Detaylı Loglama:** İşlem adımlarının takibi için loglama mekanizması.

## 🛠️ Kullanılan Teknolojiler ve Modeller

* **Büyük Dil Modeli (LLM):** `mistralai/Mistral-7B-Instruct-v0.3`
* **Embedding Modeli:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
* **Vektör Veritabanı:** `faiss-cpu`
* **Ana Kütüphaneler:**
    * `transformers`
    * `sentence-transformers`
    * `torch` (PyTorch)
    * `langchain`, `langchain_community`, `langchain_experimental`
    * `nltk` (Türkçe tokenizasyon ve stopword'ler için)
    * `pdfplumber` (PDF metin çıkarma için)
    * `rouge_score`, `bert_score`, `bleurt` (Değerlendirme metrikleri için)
    * `BitsAndBytesConfig` (4-bit quantization için)
    * `scikit-learn` (Kosinüs benzerliği için)

## ⚙️ Kurulum

1.  **Kütüphane Kurulumu:**
    * Notebook'un ilk hücresi (`Hücre 1: Gerekli Kütüphanelerin Kurulumu`) `pip install` komutları ile gerekli tüm Python kütüphanelerini kurar. Bu hücrenin eksiksiz çalıştırıldığından emin olun.

2.  **Hugging Face Girişi:**
    * `Hücre 2: Hugging Face Girişi` içerisinde, Hugging Face Hub'dan özel modelleri veya belirli modelleri indirmek için bir token ile giriş yapılır.
    * **ÖNEMLİ:** Kodda `hf_token_sabit` değişkenine atanan token size ait olmalı ve gizli tutulmalıdır. Projeyi paylaşırken bu token'ı koddan çıkarmanız veya güvenli bir şekilde (örn: Colab Secrets, ortam değişkenleri) yönetmeniz önerilir.

3.  **NLTK Kaynakları:**
    * `Hücre 3: Kütüphane İçe Aktarma, NLTK Kurulumu, BLEURT Modül Kontrolü` içerisinde NLTK için gerekli olan `wordnet`, `omw-1.4`, `punkt`, `stopwords` gibi veri paketleri indirilir.

4.  **Google Drive Bağlantısı ve Dizin Yapısı:**
    * `Hücre 4: Temel Konfigürasyonlar ve Google Drive Bağlantısı` Google Drive'ı `/content/drive` altına bağlar.
    * Proje için ana çalışma dizini `BASE_DRIVE_PATH` (varsayılan: `/content/drive/MyDrive/Colab_RAG_Projesi`) olarak ayarlanır.
    * Bu ana dizin altında `PDF_Dosyalari` (işlenecek PDF'ler için) ve `cache_data` (önbellek için) klasörleri oluşturulur veya kullanılır.
    * Eğer Google Drive bağlanamazsa, proje lokal Colab dizinlerinde çalışmaya devam eder ancak veriler kalıcı olmaz.

5.  **BLEURT Checkpoint Dosyası:**
    * BLEURT skorunun hesaplanabilmesi için `BLEURT-20.zip` checkpoint dosyasının Google Drive'ınızda `[BASE_DRIVE_PATH]/bleurt_resources/` altında bulunması gerekmektedir.
    * `Hücre 9: Cevap Kalitesi Değerlendirme Fonksiyonları` bu ZIP dosyasını Colab ortamına açarak BLEURT scorer'ını yükler.

## 🚀 Kullanım

Notebook'taki hücreler sırayla çalıştırılmalıdır.

1.  **Hazırlık Adımları (Hücre 1-5):**
    * Hücre 1: Kütüphaneleri kurar.
    * Hücre 2: Hugging Face'e giriş yapar.
    * Hücre 3: Temel kütüphaneleri içe aktarır, NLTK kaynaklarını ve BLEURT modülünü hazırlar.
    * Hücre 4: Google Drive'ı bağlar, temel dizinleri ve model ID'lerini ayarlar.
    * Hücre 5: LLM ve Embedding modellerini yükler.

2.  **Fonksiyon Tanımlamaları (Hücre 6-10):**
    * Hücre 6: PDF işleme, metin ön işleme, parçalama (chunking) ve önbellekleme için yardımcı fonksiyonları tanımlar.
    * Hücre 7: FAISS vektör indeksi oluşturma ve arama fonksiyonlarını tanımlar.
    * Hücre 8: LLM ile cevap üretme fonksiyonunu tanımlar.
    * Hücre 9: Cevap kalitesi değerlendirme metriklerini hesaplayan fonksiyonları ve BLEURT yükleyicisini tanımlar.
    * Hücre 10: Tüm RAG akışını yöneten ana `rag_pipeline` fonksiyonunu tanımlar.

3.  **PDF Seçimi ve Pipeline Çalıştırma (Hücre 11-12):**
    * **PDF Yükleme:** İşlem yapmak istediğiniz PDF dosyasını Google Drive'daki `[BASE_DRIVE_PATH]/PDF_Dosyalari/` klasörüne yükleyin.
    * **Hücre 11:** Bu hücreyi çalıştırdığınızda, `PDF_Dosyalari` klasöründeki PDF'ler listelenir. Eğer birden fazla PDF varsa, işlem yapmak istediğiniz PDF'in numarasını girmeniz istenir. Tek PDF varsa otomatik seçilir.
    * **Hücre 12:** Bu hücre, RAG pipeline için temel ayarları (chunklama metodu, chunk boyutu, retrieval top-k, LLM token sayısı vb.) tanımlar.
        * Önce varsayılan bir soru ve referans cevap ile sistemi test eder.
        * Ardından, kullanıcıya kendi sorularını sorması ve isteğe bağlı referans cevaplar girmesi için interaktif bir döngü başlatır.

## 🧩 Pipeline Adımları (Özet)

`rag_pipeline` fonksiyonu (Hücre 10) aşağıdaki adımları izler:
1.  **Önbellek Kontrolü:** İşlenecek PDF ve ayarlara göre önbellekte hazır işlenmiş veri (chunk'lar ve FAISS indeksi) olup olmadığını kontrol eder. Varsa, bu veriyi yükler.
2.  **Veri İşleme (Önbellek Yoksa):**
    * PDF'ten metin çıkarır (`pdf_to_text`).
    * Metni parçalara (chunk) ayırır (yapılandırılan metoda göre: anlamsal veya yinelemeli).
    * Parçaları ön işler (`preprocess_text`).
    * Ön işlenmiş parçalardan FAISS vektör indeksi oluşturur (`build_faiss_index`).
    * Bu işlenmiş verileri ileride kullanmak üzere önbelleğe kaydeder (`save_chunks_and_index`).
3.  **Sorgu İşleme:** Kullanıcının sorusunu ön işler.
4.  **Benzerlik Arama (Retrieval):** Ön işlenmiş soruya en uygun metin parçalarını FAISS indeksinden bulur (`retrieve_relevant_chunks`).
5.  **Bağlam Oluşturma:** Bulunan metin parçalarını birleştirerek LLM için bağlam (context) oluşturur.
6.  **Cevap Üretme (Generation):** Oluşturulan bağlam ve orijinal soru ile LLM'den cevap üretir (`generate_answer_with_llm`).
7.  **Değerlendirme:** Üretilen cevabı (eğer varsa) referans cevapla karşılaştırarak kalite metriklerini hesaplar (`evaluate_answer_quality`).
8.  **Sonuç Sunumu:** Üretilen cevabı ve değerlendirme metriklerini kullanıcıya gösterir.

## ⚙️ Yapılandırma Parametreleri (Hücre 12)

`Hücre 12` içerisinde RAG pipeline'ının davranışını etkileyen önemli parametreler bulunmaktadır:

* `RAG_USE_SEMANTIC_CHUNKER`: `True` ise anlamsal parçalayıcı, `False` ise yinelemeli parçalayıcı kullanılır.
* `RAG_RECURSIVE_CHUNK_SIZE`: Yinelemeli parçalayıcı için hedef chunk boyutu.
* `RAG_RECURSIVE_CHUNK_OVERLAP`: Yinelemeli parçalayıcı için chunk'lar arası örtüşme miktarı.
* `RAG_SEMANTIC_THRESHOLD_TYPE` / `_AMOUNT`: Anlamsal parçalayıcı için eşik değeri tipi ve miktarı.
* `RAG_RETRIEVAL_TOP_K`: LLM'e bağlam olarak sunulacak en ilgili chunk sayısı.
* `RAG_LLM_MAX_NEW_TOKENS`: LLM'in üreteceği maksimum yeni token sayısı.
* `RAG_LLM_TEMPERATURE`: LLM cevap üretimindeki rastgelelik seviyesi (düşük değerler daha deterministik sonuçlar verir).
* `RAG_LLM_DO_SAMPLE`: LLM cevap üretiminde örnekleme kullanılıp kullanılmayacağı.
* `RAG_LOG_RETRIEVED_CHUNKS_CONTENT`: Bulunan chunk'ların içeriğinin loglanıp loglanmayacağı.

## 📝 Loglama

Notebook, işlemler sırasında detaylı bilgi ve hata ayıklama logları üretmek üzere bir loglama mekanizması kullanmaya çalışır. Eğer `logger` objesi (genellikle Hücre 0'da tanımlanır, ancak bu notebook paylaşımında Hücre 0 mevcut değil) bulunamazsa, temel `print` ifadeleri kullanılır.

## 💡 Olası Geliştirmeler ve Notlar

* **Hugging Face Token Güvenliği:** Kodda sabit olarak belirtilen `hf_token_sabit` değişkeni güvenlik riski oluşturabilir. Bu token'ın Colab Secrets veya ortam değişkenleri aracılığıyla yönetilmesi önerilir.
* **Logger Başlatma:** Notebook'ta `logger` objesinin varlığı sıkça kontrol ediliyor. Eğer bu notebook bağımsız çalıştırılacaksa, en başta bir logger konfigürasyon hücresi (genellikle "Hücre 0" olarak adlandırılır) eklenmesi, logların daha düzenli ve merkezi bir şekilde yönetilmesini sağlar.
* **BLEURT Checkpoint Esnekliği:** BLEURT checkpoint yolu şu anda sabit. Farklı checkpoint'ler veya otomatik indirme mekanizmaları eklenebilir.
* **Hata Yönetimi:** Kodda birçok `try-except` bloğu mevcut, bu iyi bir pratiktir. Hata mesajları daha kullanıcı dostu hale getirilebilir veya spesifik hata türlerine göre farklı aksiyonlar alınabilir.



## 📞 İletişim

🐛 **Bug Report**: GitHub Issues kullanın  
💡 **Feature Request**: Discussions bölümünden önerinizi paylaşın  
📧 E-posta: [mehmetaksoy49@gmail.com]

- Pull Request ile katkıda bulunun
- Projeyi yıldızlamayı unutmayın! ⭐

---

**Not**: Bu proje eğitim amaçlı geliştirilmiştir ve akademik çalışmalarda referans olarak kullanılabilir.
