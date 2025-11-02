"""
إنشاء بيانات تجريبية للاختبار
Generate Example Data for Testing
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_example_data(n_samples=1000, n_features=100, output_file='example_data.csv'):
    """
    إنشاء بيانات تصنيف تجريبية
    
    Parameters:
    -----------
    n_samples : int
        عدد الصفوف
    n_features : int
        عدد الميزات
    output_file : str
        اسم ملف الإخراج
    """
    # إنشاء بيانات تصنيف
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=30,
        n_classes=3,
        random_state=42
    )
    
    # تحويل إلى DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # حفظ في ملف CSV
    df.to_csv(output_file, index=False)
    print(f"✅ تم إنشاء {output_file}")
    print(f"📊 الشكل: {df.shape}")
    print(f"🎯 الفئات: {df['target'].nunique()}")
    
    return df

if __name__ == "__main__":
    generate_example_data()

