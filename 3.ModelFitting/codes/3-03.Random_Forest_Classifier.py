# 독성 예측에 필수적인 코드만 포함하고 있습니다.
# 데이터셋 scv 파일을 같은 경로상의 디렉터리에 두고 실행해주세요.



# 라이브러리 불러오기
import pandas as pd
#1. pandas: 데이터를 빠르고 강력하게, 그리고 유연하고 쉽게 다룰 수 있게 해주는 데이터 분석, 증강 도구입니다.
#    - <https://pandas.pydata.org/docs/user_guide/index.html>

import numpy as np
#2. numpy: 파이썬에서 수치 계산을 빠르게 수행할 수 있게 해주는 도구입니다.
#    - <https://numpy.org/doc/stable/>

import matplotlib.pyplot as plt
#3. matplolib: 정적, 애니메이션, 또는 상호작용형 시각화를 생성하기 위한 포괄적인 도구입니다.
#    - <https://matplotlib.org/stable/users/index>

import sklearn
#4. scikit-learn: 데이터 분석을 위한, 쉽고 효율적인 여러 도구를 제공합니다.
#    - <https://scikit-learn.org/stable/user_guide.html>



df = pd.read_csv('tox21_train.csv')
# 이 주피터파일과 같은 디렉터리에 존재하는 'tox21_dataset.csv' 파일을 읽어와 df에 저장합니다.





X = df.iloc[:, :-1]
# 독립 변수를 X에 저장합니다. 독립 변수는 'NR-AR' 열을 제외한 나머지 열입니다.

y = df.iloc[:, -1]
# 종속 변수를 y에 저장합니다. 종속 변수는 'NR-AR' 열입니다.



from sklearn.ensemble import RandomForestClassifier
# 랜덤 포레스트 분류 모델을 불러옵니다.
import matplotlib.pyplot as plt
# 시각화를 위한 matplotlib.pyplot을 불러옵니다.
import pandas as pd
# 데이터프레임을 다루기 위한 pandas를 불러옵니다.


# 랜덤 포레스트 분류 모델 생성
model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
# 랜덤 포레스트 분류 모델을 생성합니다.
#   n_estimators: 생성할 트리의 개수입니다.
#   max_depth: 트리의 최대 깊이입니다.
#   random_state: 랜덤 시드입니다.

model.fit(X, y)
# 랜덤 포레스트 분류 모델을 학습합니다.



from sklearn.tree import plot_tree
# 트리 시각화를 위한 plot_tree 함수를 불러옵니다.

# 처음 세 개의 트리 시각화
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))  # 플롯 사이즈 조정
for index in range(0, 3):
    # 처음 세 개의 트리를 시각화합니다.
    plot_tree(model.estimators_[index], filled=True, feature_names=X.columns, class_names=["malignant", "benign"], rounded=True, ax=axes[index])
    #   forest.estimators_[index]: 랜덤 포레스트 모델의 index번째 트리를 불러옵니다.

plt.title('The first three trees of the random forest')
plt.show()
# 시각화한 트리를 출력합니다.


# 변수 중요도 계산
importances = model.feature_importances_
# 변수 중요도를 계산합니다.
indices = np.argsort(importances)
# 중요도 순으로 정렬합니다.



# 데이터프레임으로 변환
importances_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
# 중요도를 데이터프레임으로 변환합니다.



# 중요도 순으로 정렬하고 상위 10개만 선택
importances_df = importances_df.sort_values('importance', ascending=False).head(10)
# 중요도를 기준으로 내림차순 정렬하고 상위 10개만 선택합니다.



# 시각화
plt.figure(figsize=(10, 6))
# 플롯 사이즈를 조정합니다.
plt.title('Feature Importances of Random Forest Model')
# 플롯 제목을 추가합니다.
plt.barh(importances_df['feature'], importances_df['importance'], color='b', align='center')
# 막대 그래프를 생성합니다.
plt.xlabel('Relative Importance')
# x축 레이블을 추가합니다.
plt.show()
# 그래프를 출력합니다.



from sklearn.decomposition import PCA
# PCA를 위한 라이브러리를 불러옵니다.
from sklearn.metrics import confusion_matrix
# 혼동 행렬을 계산하기 위한 라이브러리를 불러옵니다.



# PCA를 사용하여 피처를 2개로 줄이기
pca = PCA(n_components=2)
# 주성분을 2개로 설정합니다.
X_pca = pca.fit_transform(X)
# PCA를 사용하여 피처를 2개로 줄입니다.



# 결정 트리 모델 학습
model_pca = RandomForestClassifier(max_depth=3, random_state=42)
# 랜덤 포레스트 분류 모델을 생성합니다.
model_pca.fit(X_pca, y)
# 랜덤 포레스트 분류 모델을 학습합니다.



# 결정 경계 시각화
x1s = np.linspace(X_pca[:, 0].min()-0.5, X_pca[:, 0].max()+0.5, 100)
# x1s: X_pca의 첫 번째 열의 최솟값에서 0.5를 뺀 값부터 최댓값에서 0.5를 더한 값까지 100개의 구간으로 나눈 배열
x2s = np.linspace(X_pca[:, 1].min()-0.5, X_pca[:, 1].max()+0.5, 100)
# x2s: X_pca의 두 번째 열의 최솟값에서 0.5를 뺀 값부터 최댓값에서 0.5를 더한 값까지 100개의 구간으로 나눈 배열
x1, x2 = np.meshgrid(x1s, x2s)
# x1, x2: x1s와 x2s를 격자로 나눈 배열
X_new = np.c_[x1.ravel(), x2.ravel()]
# X_new: x1과 x2를 1차원 배열로 변환한 후, 열로 합친 배열
y_pred = model_pca.predict(X_new).reshape(x1.shape)
# y_pred: X_new를 사용하여 예측한 후, x1의 모양으로 변환한 배열



plt.contourf(x1, x2, y_pred, alpha=0.3)
# x1, x2, y_pred를 사용하여 등고선을 그립니다.
plt.scatter(X_pca[:, 0][y==0], X_pca[:, 1][y==0], color='blue', alpha=0.1, label='NR-AR: 0')
# y가 0인 행을 산점도로 그리고, 색을 파란색으로 지정합니다.
plt.scatter(X_pca[:, 0][y==1], X_pca[:, 1][y==1], color='red', alpha=0.1, label='NR-AR: 1')
# y가 1인 행을 산점도로 그리고, 색을 빨간색으로 지정합니다.
plt.title('Decision Boundary of Random Forest Classifier with PCA')
# 그래프의 제목을 설정합니다.
plt.legend(loc='upper right')
# 범례를 표시합니다.
plt.xlabel('Principal Component 1')
# x축의 라벨을 설정합니다.
plt.ylabel('Principal Component 2')
# y축의 라벨을 설정합니다.
plt.show()
# 그래프를 출력합니다.

confusion_matrix(y, model_pca.predict(X_pca))
# 혼동 행렬을 출력합니다.

# 테스트 데이터를 학습한 모델로 분류한 결과를 혼동 행렬로 나타냅니다. 


df_test = pd.read_csv('tox21_test.csv')
# 이 주피터파일과 같은 디렉터리에 존재하는 'tox21_test.csv' 파일을 읽어와 df에 저장합니다.
X_test = df_test.iloc[:, :-1]
# 독립 변수를 X에 저장합니다. 독립 변수는 'NR-AR' 열을 제외한 나머지 열입니다.
y_test = df_test.iloc[:, -1]
# 종속 변수를 y에 저장합니다. 종속 변수는 'NR-AR' 열입니다.




# 혼동 행렬
from sklearn.metrics import confusion_matrix
# 혼동 행렬을 계산하기 위한 라이브러리를 불러옵니다.

print(confusion_matrix(y_test, model.predict(X_test)))
# 테스트 데이터의 혼동 행렬을 출력합니다.