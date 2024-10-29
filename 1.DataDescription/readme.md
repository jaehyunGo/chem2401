# Data Description
Tox21 데이터와 AC50는 독성 평가 및 약물 개발 연구에서 중요한 역할을 합니다.

1. Tox21 데이터
Tox21은 미국 환경보호청(EPA), 미국 보건복지부(DHHS), 미국 식품의약국(FDA), 미국 국립보건원(NIH) 등 여러 정부 기관이 협력하여 개발한 독성 예측을 위한 대규모 데이터셋입니다.
이 데이터셋은 다양한 화합물에 대한 독성 정보를 포함하고 있으며, 특히 신약 개발, 환경 독성 평가 등에서 화합물이 인간 건강에 미치는 영향을 예측하기 위한 목적으로 사용됩니다.
Tox21 데이터는 독성 평가를 위한 다중 분석(assays) 결과를 포함하며, 각 화합물의 화학 구조, 생물학적 특성, 독성 반응 등 다양한 변수를 제공합니다. 이를 통해 특정 화합물이 인체의 특정 시스템(예: 간, 신경계)에 미치는 영향을 평가하고, 독성 프로파일을 이해하는 데 도움을 줍니다.
2. AC50 (Half-maximal Activity Concentration)
AC50는 화합물이 반응의 절반을 유발하는 농도를 의미합니다. 여기서 "AC"는 "Activity Concentration"을 나타내며, 50은 반응의 절반에 해당하는 수치를 의미합니다.
약리학 및 독성학에서 매우 중요한 지표로, 특정 화합물이 생체에서 반응을 일으키기 위해 필요한 최소 농도를 평가하는 데 사용됩니다.
예를 들어, Tox21 데이터 내 화합물의 AC50 값은 화합물이 특정 독성 반응(예: 세포 사멸, 효소 억제)을 절반 정도 일으키는 데 필요한 농도를 나타냅니다. 낮은 AC50 값은 특정 독성 반응을 유발하는 데 필요한 농도가 낮음을 의미하며, 이는 해당 화합물이 더 높은 독성을 지닐 수 있음을 시사합니다.
이를 통해 연구자들은 화합물의 효능 및 독성을 평가하고, 최적의 농도를 설정하여 독성 위험을 최소화할 수 있습니다.
Tox21 데이터와 AC50은 함께 사용되어 화합물의 독성 예측, 약물 안전성 평가, 그리고 환경 독성 연구 등에서 매우 중요한 역할을 하며, 특히 독성 평가 모델 및 머신러닝 알고리즘의 훈련에 유용하게 활용됩니다.

고재현

**2. Data Preprocessing**

* 2-01. tox21_EDA


범주형 데이터를 가지는 tox21.xlsm 데이터셋 분석

* 2-02. tox21_ac50_EDA


(보충) 연속형 데이터를 가지는 tox21_ac50.xlsx 데이터셋 분석

* 2-03. tox21_Preprocessing


tox21.xlsm 전처리 진행

---

**3. Model Fitting**


아래의 10가지 모델을 통해 독성 유무 예측


* 3-01.Logistic_Regression


* 3-02.Decision_Tree_Classifier


* 3-03.Random_Forest_Classifier 


* 3-04.Gradient_Boosting_Classifier 


* 3-05.XGB_Classifier


* 3-06.LGBM_Classifier


* 3-07.Linear_Discriminant_Analysis


* 3-08.Quadratic_Discriminant_Analysis


* 3-09.PLS_Regression


* 3-10.MLP_Classifier

---

**4. Model Evaluation**


평가지표와 교차 검증을 통해 모델 성능 평가


* 4-01. Evaluation
