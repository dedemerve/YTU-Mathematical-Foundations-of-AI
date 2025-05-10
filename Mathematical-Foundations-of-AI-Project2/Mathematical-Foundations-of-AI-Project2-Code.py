import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import os
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

def get_positive_int(prompt_message):
    """Kullanıcıdan pozitif tam sayı al"""
    while True:
        try:
            value = int(input(prompt_message))
            if value <= 0:
                print("Lütfen pozitif bir tam sayı girin!")
                continue
            return value
        except ValueError:
            print("Geçersiz giriş! Lütfen pozitif bir tam sayı girin.")

def load_dataset(dosya_yolu):
    """Veri setini yükle ve ön işleme"""
    try:
        # Dosya yolundaki tırnak işaretlerini temizle
        dosya_yolu = dosya_yolu.strip('"').strip("'")
        
        # Dosyanın var olup olmadığını kontrol et
        if not os.path.isfile(dosya_yolu):
            raise FileNotFoundError(f"Dosya bulunamadı: {dosya_yolu}")
        
        # Önce dosyayı oku ve içeriğine bak
        with open(dosya_yolu, 'r') as f:
            ilk_satir = f.readline().strip()
            print("\nDosyanın ilk satırı:")
            print(ilk_satir)
            
            # Olası ayırıcıları belirle
            virgul_sayisi = ilk_satir.count(',')
            bosluk_sayisi = ilk_satir.count(' ')
            tab_sayisi = ilk_satir.count('\t')
            noktali_virgul_sayisi = ilk_satir.count(';')
            
            # En çok kullanılan ayırıcıyı seç
            ayiricilar = {
                ',': virgul_sayisi,
                ' ': bosluk_sayisi,
                '\t': tab_sayisi,
                ';': noktali_virgul_sayisi
            }
            en_cok_kullanilan = max(ayiricilar.items(), key=lambda x: x[1])
            secilen_ayirici = en_cok_kullanilan[0]
            
            print(f"\nSeçilen ayırıcı: '{secilen_ayirici}'")
        
        # Veriyi seçilen ayırıcı ile oku
        try:
            veri = pd.read_csv(dosya_yolu, delimiter=secilen_ayirici)
        except:
            # Eğer başarısız olursa diğer ayırıcıları dene
            for ayirici in [',', ' ', '\t', ';']:
                if ayirici != secilen_ayirici:
                    try:
                        veri = pd.read_csv(dosya_yolu, delimiter=ayirici)
                        if len(veri.columns) > 1:
                            print(f"\nBaşarılı ayırıcı: '{ayirici}'")
                            break
                    except:
                        continue
        
        print(f"\nVeri seti boyutu: {veri.shape}")
        print("\nVeri setinin ilk birkaç satırı:")
        print(veri.head())
        
        # Sütun sayısını kontrol et
        if veri.shape[1] == 1:
            raise ValueError("Veri tek sütun olarak okundu. Ayırıcı karakter problemi olabilir.")
        
        # Son sütunun hedef değişken olduğunu varsayıyoruz
        X = veri.iloc[:, :-1].values
        y = veri.iloc[:, -1].values
        
        # Özellik isimlerini al (varsa)
        ozellik_isimleri = veri.columns[:-1].tolist()
        if not ozellik_isimleri:
            ozellik_isimleri = [f'Ozellik_{i}' for i in range(X.shape[1])]
        
        return X, y, ozellik_isimleri
        
    except Exception as e:
        print(f"\nVeri yükleme hatası: {str(e)}")
        raise

def calculate_fidelity(teacher_predictions, student_predictions):
    """Model sadakati (fidelity) hesapla"""
    return accuracy_score(teacher_predictions, student_predictions)

def calculate_metrics(y_true, y_pred):
    """Tüm metrikleri hesapla"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics

def save_metrics_report(normal_metrics_train, normal_metrics_test, 
                       pruned_metrics_train, pruned_metrics_test,
                       fidelity_score, wilcoxon_result, output_file="metrics_report.txt"):
    """Metrikleri dosyaya kaydet"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Başlık
        f.write("Karar Ağacı Analiz Raporu\n")
        f.write("========================\n\n")
        
        # Tablo formatında metrikler
        f.write("Detaylı Metrik Tablosu:\n")
        f.write("-" * 80 + "\n")
        header = "|{:^20}|{:^14}|{:^14}|{:^14}|{:^14}|".format(
            "Metrik", "Normal-Train", "Normal-Test", "Budanmış-Train", "Budanmış-Test"
        )
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        # Metrik değerlerini tabloya ekle
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            row = "|{:^20}|{:^14.6f}|{:^14.6f}|{:^14.6f}|{:^14.6f}|".format(
                metric,
                normal_metrics_train[metric],
                normal_metrics_test[metric],
                pruned_metrics_train[metric],
                pruned_metrics_test[metric]
            )
            f.write(row + "\n")
        f.write("-" * 80 + "\n\n")
        
        # Detaylı sayısal değerler
        f.write("Detaylı Sayısal Değerler:\n")
        f.write("========================\n\n")
        
        f.write("1. Normal Karar Ağacı Metrikleri\n")
        f.write("--------------------------\n")
        f.write("Eğitim Seti:\n")
        for metric, value in normal_metrics_train.items():
            f.write(f"{metric}: {value:.6f}\n")
        
        f.write("\nTest Seti:\n")
        for metric, value in normal_metrics_test.items():
            f.write(f"{metric}: {value:.6f}\n")
        
        f.write("\n2. Budanmış Karar Ağacı Metrikleri\n")
        f.write("-----------------------------\n")
        f.write("Eğitim Seti:\n")
        for metric, value in pruned_metrics_train.items():
            f.write(f"{metric}: {value:.6f}\n")
        
        f.write("\nTest Seti:\n")
        for metric, value in pruned_metrics_test.items():
            f.write(f"{metric}: {value:.6f}\n")
        
        f.write("\n3. Model Karşılaştırma Metrikleri\n")
        f.write("---------------------------\n")
        f.write(f"Fidelity Score: {fidelity_score:.6f}\n")
        f.write(f"Wilcoxon Test İstatistiği: {wilcoxon_result.statistic:.6f}\n")
        f.write(f"Wilcoxon p-değeri: {wilcoxon_result.pvalue:.6f}\n")

def find_optimal_depth(X_train, X_test, y_train, y_test, max_depth_range=range(1, 21)):
    """En iyi ağaç derinliğini bul"""
    
    train_scores = []
    test_scores = []
    
    for depth in max_depth_range:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = accuracy_score(y_train, dt.predict(X_train))
        test_score = accuracy_score(y_test, dt.predict(X_test))
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_range, train_scores, label='Eğitim Doğruluğu', marker='o')
    plt.plot(max_depth_range, test_scores, label='Test Doğruluğu', marker='o')
    plt.xlabel('Maksimum Derinlik')
    plt.ylabel('Doğruluk')
    plt.title('Ağaç Derinliğine Göre Model Performansı')
    plt.legend()
    plt.grid(True)
    plt.savefig('depth_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    best_depth = max_depth_range[np.argmax(test_scores)]
    print(f"\nEn iyi test doğruluğu derinlik {best_depth}'de elde edildi")
    print(f"Test Doğruluğu: {max(test_scores):.4f}")
    
    return best_depth

def analyze_decision_tree(X_train, X_test, y_train, y_test, max_depth=None):
    """Genişletilmiş karar ağacı analizi"""
    
    # Normal karar ağacı
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    
    # Normal ağaç tahminleri
    train_pred_normal = dt.predict(X_train)
    test_pred_normal = dt.predict(X_test)
    
    # Normal ağaç metrikleri
    normal_metrics_train = calculate_metrics(y_train, train_pred_normal)
    normal_metrics_test = calculate_metrics(y_test, test_pred_normal)
    
    # Budanmış karar ağacı
    pruned_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    pruned_dt.fit(X_train, y_train)
    
    # Budanmış ağaç tahminleri
    train_pred_pruned = pruned_dt.predict(X_train)
    test_pred_pruned = pruned_dt.predict(X_test)
    
    # Budanmış ağaç metrikleri
    pruned_metrics_train = calculate_metrics(y_train, train_pred_pruned)
    pruned_metrics_test = calculate_metrics(y_test, test_pred_pruned)
    
    # Fidelity hesapla
    fidelity_train = calculate_fidelity(train_pred_normal, train_pred_pruned)
    
    # Wilcoxon testi
    wilcoxon_result = wilcoxon(test_pred_normal, test_pred_pruned)
    
    # Metrikleri kaydet
    save_metrics_report(
        normal_metrics_train, normal_metrics_test,
        pruned_metrics_train, pruned_metrics_test,
        fidelity_train, wilcoxon_result
    )
    
    # 1. Metrik Karşılaştırma Grafiği
    plt.figure(figsize=(12, 6))
    metrics_data = pd.DataFrame({
        'Metrik': list(normal_metrics_test.keys()) * 2,
        'Model': ['Normal'] * 4 + ['Budanmış'] * 4,
        'Değer': list(normal_metrics_test.values()) + list(pruned_metrics_test.values())
    })
    sns.barplot(x='Metrik', y='Değer', hue='Model', data=metrics_data)
    plt.title('Model Metrik Karşılaştırması (Test Seti)')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Özellik Önem Grafiği
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'Özellik': range(X_train.shape[1]),
        'Önem': pruned_dt.feature_importances_
    }).sort_values('Önem', ascending=False)
    sns.barplot(x='Özellik', y='Önem', data=feature_importance)
    plt.title('Özellik Önem Sıralaması')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Karar Ağacı Görselleştirmesi
    plt.figure(figsize=(20,10))
    plot_tree(pruned_dt, feature_names=[f'Özellik_{i}' for i in range(X_train.shape[1])], 
             class_names=[str(i) for i in np.unique(y_train)],
             filled=True, rounded=True)
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dt, pruned_dt

def main():
    print("Karar Ağacı Analiz Programı")
    print("-----------------------")
    
    try:
        # Veri setini yükle ve böl
        dosya_yolu = input("Lütfen veri seti dosya yolunu girin: ")
        X, y, ozellik_isimleri = load_dataset(dosya_yolu)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nVeri {len(X_train)} eğitim ve {len(X_test)} test örneği olarak bölündü")
        
        # Otomatik optimizasyon seçeneği sun
        while True:
            secim = input("\nAğaç derinliğini otomatik optimize etmek ister misiniz? (e/h): ").lower()
            if secim in ['e', 'h']:
                break
            print("Lütfen 'e' veya 'h' girin!")
        
        if secim == 'e':
            print("\nOptimal derinlik aranıyor...")
            max_derinlik = find_optimal_depth(X_train, X_test, y_train, y_test)
        else:
            max_derinlik = get_positive_int("\nKarar ağacı için maksimum derinlik girin: ")
        
        # Karar ağaçlarını analiz et
        dt, pruned_dt = analyze_decision_tree(X_train, X_test, y_train, y_test, max_depth=max_derinlik)
        
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nBir hata oluştu: {str(e)}")
    finally:
        print("\nProgram tamamlandı.")

if __name__ == "__main__":
    main()