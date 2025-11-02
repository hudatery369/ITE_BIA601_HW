# مشروع اختيار الميزات باستخدام الخوارزميات الجينية

**إشراف:** د. عصام سلمان

**الطلاب:**
- أسامة الحوري
- روان طلال الفيحان
- هدى محمد طيري
- أحمد علاوي
- حسن
- طارق نصر 
- يزن
- أحمد

---

## التثبيت والتشغيل

### 1. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 2. تشغيل التطبيق
```bash
python app.py
```

### 3. فتح المتصفح
افتح: `http://localhost:5000`

---

## الاستخدام

1. حمّل ملف البيانات (CSV أو Excel)
2. اختر عمود الهدف (Target)
3. اضبط معاملات الخوارزمية الجينية
4. اضغط "تشغيل الخوارزمية الجينية"
5. شاهد النتائج

---

## البنية

```
genetic-feature-selection/
├── app.py                    # نقطة البداية
├── requirements.txt          # المتطلبات
├── README.md                 # الدليل
│
├── genetic_algorithm/        # الخوارزمية الجينية
│   ├── chromosome.py         # تمثيل الكروموسوم
│   ├── fitness.py            # حساب اللياقة
│   ├── operators.py          # العمليات الجينية
│   └── ga_engine.py          # المحرك الرئيسي
│
├── data_processing/          # معالجة البيانات
│   ├── loader.py             # تحميل CSV/Excel
│   ├── preprocessor.py       # تنظيف وتطبيع البيانات
│   └── validator.py          # التحقق من البيانات
│
├── models/                   # نماذج التعلم الآلي
│   ├── ml_models.py          # Random Forest
│   ├── evaluator.py          # تقييم النموذج
│   └── metrics.py            # مقاييس الأداء
│
├── comparison/               # الطرق التقليدية
│   └── traditional_methods.py # ANOVA, MI, RF Importance
│
├── web/                      # واجهة الويب
│   ├── routes.py             # مسارات Flask
│   └── templates/
│       └── index.html        # الصفحة الرئيسية
│
├── sample_datasets/          # بيانات تجريبية
│   ├── iris.csv
│   ├── wine.csv
│   └── breast_cancer.csv
│
├── uploads/                  # ملفات المستخدم المحملة
└── results/                  # نتائج التجارب
    └── experiments/
```

