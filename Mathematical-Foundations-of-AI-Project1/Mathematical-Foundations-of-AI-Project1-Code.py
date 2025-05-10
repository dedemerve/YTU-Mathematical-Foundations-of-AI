import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from tkinter import filedialog
import tkinter as tk

# Dosya seçimi için tkinter penceresi
root = tk.Tk()
root.withdraw()

# Dosya seçme diyaloğu
file_path = filedialog.askopenfilename(title="CSV dosyasını seçin", 
                                     filetypes=[("CSV files", "*.csv")])

if file_path:
    # Veri setini okuma
    data = pd.read_csv(file_path)
    
    # Verilerin ilk 5 satırını göster
    print("Veri seti başlangıcı:")
    print(data.head())
    print("\nVeri seti boyutu:", data.shape)
    print("\nEksik veriler:")
    print(data.isnull().sum())
    
    # Kategorik değişkenleri dönüştürme
    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    
    # Veriyi ayırma
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']
    
    # Ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Veri setini bölme
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Linear SVM
    linear_svm = SVC(kernel='linear', random_state=42)
    linear_svm.fit(X_train, y_train)
    linear_pred = linear_svm.predict(X_test)
    
    # RBF SVM
    rbf_svm = SVC(kernel='rbf', random_state=42)
    rbf_svm.fit(X_train, y_train)
    rbf_pred = rbf_svm.predict(X_test)
    
    # PCA dönüşümü
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # PCA verisi için modeller
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    linear_svm_pca = SVC(kernel='linear', random_state=42)
    linear_svm_pca.fit(X_train_pca, y_train_pca)
    
    rbf_svm_pca = SVC(kernel='rbf', random_state=42)
    rbf_svm_pca.fit(X_train_pca, y_train_pca)
    
    # Karar sınırı görselleştirme
    def plot_decision_boundary(X, y, model, title):
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title(title)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()
    
    # Görselleştirmeler
    print("\nKarar sınırları görselleştiriliyor...")
    plot_decision_boundary(X_pca, y, linear_svm_pca, 'Linear SVM Decision Boundary')
    plot_decision_boundary(X_pca, y, rbf_svm_pca, 'RBF SVM Decision Boundary')
    
    # Confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(confusion_matrix(y_test, linear_pred), annot=True, fmt='d', ax=ax1)
    ax1.set_title('Linear SVM Confusion Matrix')
    sns.heatmap(confusion_matrix(y_test, rbf_pred), annot=True, fmt='d', ax=ax2)
    ax2.set_title('RBF SVM Confusion Matrix')
    plt.show()
    
    # Model performansları
    print("\nLinear SVM Performance:")
    print(classification_report(y_test, linear_pred))
    print("\nRBF SVM Performance:")
    print(classification_report(y_test, rbf_pred))
    
    # Özellik önemliliği
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(linear_svm.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance (Linear SVM)')
    plt.show()
    
else:
    print("Dosya seçilmedi!")