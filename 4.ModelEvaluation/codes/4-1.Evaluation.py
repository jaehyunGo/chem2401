# 성능 지표에 관한 코드
from sklearn.metrics import (
	recall_score,
	precision_score,
	f1_score
)

y_true = [0, 1, 1, 0, 1, 0, 0, 1, 0]
y_predict = [0, 1, 1, 0, 1, 1, 0, 1, 1]

rec_score = recall_score(y_true, y_predict)
print("recall score:", rec_score)
prec_score = precision_score(y_true, y_predict)
print("precision score:", prec_score)

my_f1_score = 2 * (rec_score * prec_score) / (rec_score + prec_score)
print("my_f1_score:", my_f1_score)
print("f1_score:",f1_score(y_true, y_predict))

# ===============================================================
# K-fold Cross Validation에 관한 코드
# Import

# 표 형식의 데이터 등 다양한 데이터를 쉽게 다룰 수 있도록(데이터 결합 등) 해주는 모듈
import pandas as pd

# 다차원 배열을 쉽게 처리하고 효율적으로 사용할 수 있도록 해주는 모듈
import numpy as np

# 범주형 변수 예측에 사용되는 모델 중 하나
# 분류 문제(0인지 1인지 분류처럼)에 사용되는 기계학습 알고리즘 중 하나
# 현재 예시에서 사용할 모델
from sklearn.linear_model import LogisticRegression

# scikit-learn에서 제공하는 K-Fold
from sklearn.model_selection import KFold

# 사용할 데이터가 들어있는 폴더
folder = '../../0.Data'
# 사용할 데이터의 파일명
train_data_file = 'tox21_train.csv'
# ../../0.Data/tox21_train.csv과 같이 파일 경로로 만듦
train_file_path = f'{folder}/{train_data_file}'

# 파일 경로에 해당하는 csv 파일 불러오기
# csv 파일이란 comma-separated values로, 쉼표로 구분된 데이터 파일임
df = pd.read_csv(train_file_path)
print(df.shape)

# data에서 feature 부분과 label 부분 분리
X = df.iloc[:, :166]
y = df.iloc[:, 166]


# ================My K-fold================

# k-fold에서 데이터를 몇 개의 fold로 나눌 지 정함
k = 5

# feature의 데이터 개수를 저장
size = len(X)

# feature를 k개의 fold로 나누기 위해 몇 개의 데이터 단위로 끊어야 하는 지 계산
fold_size = (int)(size / k)

# 각 실험에 대한 accuracy를 저장하기 위한 배열
accuracies = np.empty(k)

# fold 개수만큼 실험 반복
for i in range(k):
	# [arange 함수 참고]
	# numpy 모듈의 arange 함수는 인자로 들어온 범위만큼의 등차수열 배열을 반환
	# 사용 -> np.arange(start, end, step) (start와 step을 지정해주지 않으면 각각 0, 1이 기본값으로 적용)
	# ex. np.arange(0, 10, 1) -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	# ex. np.arange(10) -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	# [concatenate 함수 참고]
	# numpy 모듈의 concatenate 함수는 인자로 들어온 배열을 합쳐서 새로운 배열을 반환
	# ex. np.concatenate((np.array([1, 2, 3]), np.array([4, 5, 6]))) -> [1, 2, 3, 4, 5, 6]

	# 분할할 train data & validation data의 index 저장
		# feature 중에서 validation data로 사용할 부분의 index 저장
		# 즉, validation data로 사용할 fold의 index를 저장
	val_index = np.arange(i * fold_size, (i + 1) * fold_size)
		# validation data로 사용할 부분을 제외한 부분의 index를 합쳐서 저장
		# train data로 사용할 부분의 index임
	train_index = np.concatenate((np.arange(i * fold_size), np.arange((i + 1) * fold_size, size)))

	# train data & validation data 생성
	# X 중에서 위에서 지정한 index에 해당하는 값을 train과 validation에 나누어 저장
	X_train = X.iloc[train_index]
	X_val = X.iloc[val_index]
	y_train = y.iloc[train_index]
	y_val = y.iloc[val_index]

	# 사용할 모델 지정(Logistic Regression 모델 사용)
	model = LogisticRegression()

	# 모델에 학습
	model.fit(X_train, y_train)

	# 매 실험마다 계산한 정확도를 accuracies 배열에 하나씩 저장
	# feature_test의 test data에 대해 위에서 학습한 모델이 추측한 결과와 실제 정답인 label_test를 비교하여 accuracy 계산
	accuracies[i] = model.score(X_val, y_val)

# 위에서 계산한 전체 accuracy 출력
print("My K-fold Accuracies:", accuracies)

# 모든 accuracy의 평균을 계산하여 최종 accuracy 출력
print("My K-fold Mean Accuracy:", accuracies.mean())

# ===============================================================
# ================Scikit K-fold================
# [방법 1]

# k-fold에서 데이터를 몇 개의 fold로 나눌 지 정함
k = 5

# k개의 fold로 cross validation하는 K-Fold 객체 생성
kfold = KFold(n_splits = k)

# 사용할 모델 지정(Logistic Regression 모델 사용)
model = LogisticRegression()

# 각 실험에 대한 accuracy를 저장하기 위한 배열
accuracies = np.empty(k)

# k번 만큼 실험을 반복하며, 매 회차마다의 train 데이터를 다시 train과 validation data 범위의 index를 각각 train_index와 val_index에 저장
for train_index, val_index in kfold.split(X, y):
	# validation index 범위 출력해서 확인
	print(val_index)
	
	# train data & validation data 생성
	# X 중에서 위에서 지정한 index에 해당하는 값을 train과 val 나누어 저장
	X_train, y_train = X.iloc[train_index], y.iloc[train_index]
	X_val, y_val = X.iloc[val_index], y.iloc[val_index]

	# 모델에 학습
	model.fit(X_train, y_train)

	# 매 실험마다 계산한 정확도를 accuracies 배열에 하나씩 저장
	# X_val의 validation data에 대해 위에서 학습한 모델이 추측한 결과와 실제 정답인 y_val를 비교하여 accuracy 계산
	accuracies[i] = model.score(X_val, y_val)


# 위에서 계산한 전체 accuracy 출력
print("Scikit-learn K-fold Accuracies:", accuracies)

# 모든 accuracy의 평균을 계산하여 최종 accuracy 출력
print("Scikit-learn K-fold Mean Accuracy:", accuracies.mean())

# ===============================================================
# [방법 2]

# cross validation에서 score를 계산하는 함수
from sklearn.model_selection import cross_val_score

# k-fold에서 데이터를 몇 개의 fold로 나눌 지 정함
k = 5

# k개의 fold로 cross validation하는 K-Fold 객체 생성
kfold = KFold(n_splits = k)

# 사용할 모델 지정(Logistic Regression 모델 사용)
model = LogisticRegression()

# 사용할 model, X, y, 사용할 cross validation 객체로 cross validation 했을 때의 accuracy 배열을 results에 저장
results = cross_val_score(model, X, y, cv = kfold)

# 전체 accuracy 출력
print("Scikit-Learn K-fold Accuracies:", results)

# 모든 accuracy의 평균을 계산하여 최종 accuracy 출력
print("Scikit-Learn K-fold Mean Accuracy:", results.mean())

# ===============================================================
# Stratified K-fold Cross Validation에 관한 코드
# Stratified K-Fold
from sklearn.model_selection import StratifiedKFold

# k-fold에서 데이터를 몇 개의 fold로 나눌 지 정함
k = 5

# k개의 fold로 cross validation하는 Stratified K-Fold 객체 생성
strkfold = StratifiedKFold(n_splits = k)

# 사용할 모델 지정(Logistic Regression 모델 사용)
model = LogisticRegression(max_iter=1000)

# 각 실험에 대한 accuracy를 저장하기 위한 배열
accuracies = np.empty(k)

# 반복문이 돌아가면서 accuracies 배열에 값을 저장하기 위한 index 변수
i = 0

# k번 만큼 실험을 반복하며, 매 회차마다의 train과 validation data 범위의 index를 각각 train_index와 val_index에 저장
for train_index, val_index in strkfold.split(X, y):
	# val index 범위 출력해서 확인
	print(val_index)
	
	# test data & train data 생성
	# X 중에서 위에서 지정한 index에 해당하는 값을 train과 test에 나누어 저장
	X_train, X_val = X.iloc[train_index], X.iloc[val_index]
	y_train, y_val = y.iloc[train_index], y.iloc[val_index]

	# 모델에 학습
	model.fit(X_train, y_train)

	# 매 실험마다 계산한 정확도를 accuracies 배열에 하나씩 저장
	# X_test의 test data에 대해 위에서 학습한 모델이 추측한 결과와 실제 정답인 y_test를 비교하여 accuracy 계산
	accuracies[i] = model.score(X_val, y_val)
	
	i = i + 1


# 위에서 계산한 전체 accuracy 출력
print("Stratified K-fold Accuracies:", accuracies)

# 모든 accuracy의 평균을 계산하여 최종 accuracy 출력
print("Stratified K-fold Mean Accuracy:", accuracies.mean())

# ===============================================================
# tox21 data 평가 시작

import numpy as np

# 표 형식의 데이터 등 다양한 데이터를 쉽게 다룰 수 있도록(데이터 결합 등) 해주는 모듈
import pandas as pd

# Stratified K-Fold
from sklearn.model_selection import StratifiedKFold

# 첫 번째 모델
from sklearn.linear_model import LogisticRegression

# 두 번째 모델
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
	accuracy_score,
	roc_auc_score,
	recall_score,
	precision_score,
	f1_score
)

from tqdm import tqdm

from itertools import product
from collections.abc import Iterable

import warnings
warnings.filterwarnings('ignore')

# pandas dataframe의 모든 열을 출력하도록 설정
pd.set_option('display.max_columns', None)

# train과 test csv 파일 불러오기
# csv 파일이란 comma-separated values로, 쉼표로 구분된 데이터 파일임
df = pd.read_csv('../../0.Data/tox21_train.csv')
df_test = pd.read_csv('../../0.Data/tox21_test.csv')

# df 출력
df
# df_test 출력
df_test

# df과 df_test에서 feature(X) 부분과 label(y) 부분 분리
X = df.iloc[:, :166]
y = df.iloc[:, 166]
X_test = df_test.iloc[:, :166]
y_test = df_test.iloc[:, 166]

# 반복 시행 시 같은 난수를 추출하기 위해 seed값 지정
seed = 42

# ===============================================================
# Logistic Regression 모델
logistic_params_dict = {
	# 규제의 강도를 설정(1이 기본값)
	# 값이 작을수록 규제가 강해짐
	'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 50, 100],
	
	# 규제의 종류를 설정
	# l1: Lasso 규제
	# l2: Ridge 규제
	'penalty': ['l1', 'l2'],

	# 최적화에 사용할 알고리즘 설정(lbfgs가 기본값)
	# liblinear: 작은 데이터셋에 적합한 최적화 알고리즘
	# saga: 대용량 데이터셋에 적합한 최적화 알고리즘
	'solver': ['liblinear', 'saga']
}

# logistic_params_dict의 자료형이 딕셔너리인지 확인
if not isinstance(logistic_params_dict, dict):
	# logistic_params_dict가 딕셔너리가 아니면 에러 출력
	raise TypeError('Parameter grid is not a dict ({!r})'.format(logistic_params_dict))

if isinstance(logistic_params_dict, dict):
	# logistic_params_dict의 각 원소가 반복 가능(iterable)한지 확인
	# 여기서 반복 가능하다는 것은 리스트, 튜플, 세트, 문자열 등을 의미
	for key in logistic_params_dict:
		# logistic_params_dict가 딕셔너리가 아니면 에러 출력
		if not isinstance(logistic_params_dict[key], Iterable):
			raise TypeError('Parameter grid value is not iterable '
							'(key={!r}, value={!r})'.format(key, logistic_params_dict[key]))

# 딕셔너리의 key의 알파벳 순으로 정렬
items = sorted(logistic_params_dict.items())

# 다음과 같이 key는 key끼리, value는 value끼리 모아서 저장
# item 앞에 *를 붙이면 item의 iterable한 객체들을 분리하여 인자로 전달함
# ('C', 'penalty', 'solver')
# ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 50, 100], ['l1', 'l2'], ['liblinear', 'saga'])
keys, values = zip(*items)

# 조합된 parameter를 담을 list 선언
logistic_params_grid = []

# product 함수는 iterable한 객체의 모든 가능한 조합을 생성함
# value 앞에 *를 붙이면 value의 iterable한 객체들을 분리하여 인자로 전달함
# (0.1, 'l1', 'liblinear')처럼...
for v in product(*values):
	# {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'} 이런 식으로 만들어서 하나씩 추가
	logistic_params_grid.append(dict(zip(keys, v))) 

# parameter 조합 출력
logistic_params_grid

# parameter 조합 개수 출력
print(len(logistic_params_grid))

# 결과 딕셔너리 만들기
result = {}
result['model'] = {}
result['f1'] = {}

# result 출력해보기
result

# 모든 logistic_params_dict의 조합에 대해 반복
# tqdm은 프로그램 진행 상황을 시각적으로 보여줌
for p in tqdm(range(len(logistic_params_grid))):
	result['model']['model'+str(p)] = logistic_params_grid[p]
	result['f1']['model'+str(p)] = []
	
	# logistic_params_grid[p] 딕셔너리를 인자로 넘겨줄 때 ** 사용
	# **을 사용하면 딕셔너리의 키-값 쌍을 풀어서 인자로 전달 가능
	# ex. LogisticRegression(random_state=seed, C=1.0, penalty='l2', solver='lbfgs')
	model = LogisticRegression(random_state = seed, **logistic_params_grid[p])

	# Stratified K-fold 객체 생성
	skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
	# validation f1_score 저장할 리스트 생성
	val_f1 = []

	# k번 만큼 실험을 반복하며, 매 회차마다의 train 데이터를 다시 train과 validation data 범위의 index를 각각 train_index와 val_index에 저장
	for train_idx, val_idx in skf.split(X, y):
		# train data와 validation data 생성
		X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
		X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

		# 모델에 학습
		model.fit(X_train, y_train)

		# X_val 데이터로 예측한 결과 저장
		val_predict = model.predict(X_val)
	
		# val_predict의 f1_score를 하나씩 저장
		val_f1.append(f1_score(y_val, val_predict))
	
	# 각 모델 별 validation f1_score의 평균을 저장
	result['f1']['model'+str(p)].append(np.mean(val_f1))

# 결과 출력
result

# best parameter 찾기

# 모델의 f1_score로만 list 생성
f1_list = list(map(lambda x: np.mean(x[1]), result['f1'].items()))
# f1_score가 최대인 곳의 index 저장
f1_max_idx = f1_list.index(max(f1_list))

# best parameter 저장
best_param = result['model'][f'model{f1_max_idx}']

# best_param이 위치한 index 찾기
m = list(result['model'].keys())[list(result['model'].values()).index(best_param)]

# val result 출력

# best parameter인 곳의 validation f1_score 저장
f1 = result['f1'][m]

# best parameter와 validation의 f1_score 출력
print(f"best param: {best_param} \
		\n[validation result] \
		\nf1: {f1[0]:.3f}")

# test result 출력

# 최종으로 나온 best parameter의 Logistic Regression 모델
model = LogisticRegression(random_state = seed, **best_param)


# Stratified K-fold 객체 생성
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)

# k번 만큼 실험을 반복하며, 매 회차마다의 train 데이터를 다시 train과 validation data 범위의 index를 각각 train_index와 val_index에 저장
for train_idx, val_idx in skf.split(X, y):
	# train data와 validation data 생성
	X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
	X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

	# tox21_train 데이터로 학습
	model.fit(X_train, y_train)

# tox21_test 데이터로 예측
pred = model.predict(X_test)
pred_score = model.predict_proba(X_test)[:, 1]

# best parameter와 예측 결과의 성능 지표 출력
print(f'[test result] \
		\nbest param: {best_param} \
		\nprecision: {precision_score(y_test, pred):.3f} \
		\nrecall: {recall_score(y_test, pred):.3f} \
		\naccuracy: {accuracy_score(y_test, pred):.3f} \
		\nauc: {roc_auc_score(y_test, pred_score):.3f} \
		\nf1: {f1_score(y_test, pred):.3f}')

# ===============================================================
# MLP 모델

# MLP의 hyperparameter
mlp_params_dict = {
	# 모델에 있는 은닉층의 크기와 개수
	'hidden_layer_sizes': [(50), (100, 50, 10)],
	
	# 뉴런의 출력을 결정하는 활성화 함수 지정
	# relu: ReLU 함수, max(x, 0)과 동일
	# tanh: hyperbolic tangent 함수
	'activation': ['relu', 'tanh'],
	
	# 최적화에 사용될 알고리즘
	# adam: Adaptive Moment Estimation(Adam), 이전 그래디언트의 지수적인 이동 평균을 사용하여 학습률을 조절하는 방식으로 모델을 업데이트
	# sgd: Stochastic Gradient Descent(SGD), 데이터를 무작위로 선정하여 경사 하강법으로 매개변수를 갱신하는 방법
	'solver': ['adam', 'sgd'],
	
	# L2 정규화 항의 가중치
	'alpha': [0.0001, 0.001],
	
	# 학습률의 초기값 설정
	'learning_rate_init': [0.001, 0.01],
	
	# 최대 학습 과정 반복 횟수
	'max_iter': [50, 100]
}

# mlp_params_dict의 자료형이 딕셔너리인지 확인
if not isinstance(mlp_params_dict, dict):
	# mlp_params_dict가 딕셔너리가 아니면 에러 출력
	raise TypeError('Parameter grid is not a dict ({!r})'.format(mlp_params_dict))

# mlp_params_dict가 dictionary라면
if isinstance(mlp_params_dict, dict):
	# mlp_params_dict의 각 원소가 반복 가능(iterable)한지 확인
	# 여기서 반복 가능하다는 것은 리스트, 튜플, 세트, 문자열 등을 의미
	for key in mlp_params_dict:
		# mlp_params_dict가 딕셔너리가 아니면 에러 출력
		if not isinstance(mlp_params_dict[key], Iterable):
			raise TypeError('Parameter grid value is not iterable '
							'(key={!r}, value={!r})'.format(key, mlp_params_dict[key]))

# 딕셔너리를 key의 알파벳 순으로 정렬
items = sorted(mlp_params_dict.items())

# 다음과 같이 key는 key끼리, value는 value끼리 모아서 저장
# item 앞에 *를 붙이면 item의 iterable한 객체들을 분리하여 인자로 전달함
# ('activation', 'alpha', 'hidden_layer_sizes', 'learning_rate_init', 'max_iter', 'solver')
# (['relu', 'tanh'], [0.0001, 0.001], [50, (100, 50, 10)], [0.001, 0.01], [50, 100], ['adam', 'sgd'])
keys, values = zip(*items)

# 조합된 parameter를 담을 list 선언
mlp_params_grid = []

# product 함수는 iterable한 객체의 모든 가능한 조합을 생성함
# value 앞에 *를 붙이면 value의 iterable한 객체들을 분리하여 인자로 전달함
# ('relu', 0.0001, 50, 0.001, 50, 'adam')처럼...
for v in product(*values):
	# {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': 50, 'learning_rate_init': 0.001, 'max_iter': 50, 'solver': 'adam'}
	# 이런 식으로 만들어서 하나씩 추가
	mlp_params_grid.append(dict(zip(keys, v))) 

# parameter 조합 출력
mlp_params_grid

# 가능한 parameter 조합 개수 출력
print(len(mlp_params_grid))

# 결과 딕셔너리 만들기
result = {}
result['model'] = {}
result['f1'] = {}

# result 출력해보기
result

# 모든 mlp_params_dict의 조합에 대해 반복
# tqdm은 프로그램 진행 상황을 시각적으로 보여줌
for p in tqdm(range(len(mlp_params_grid))):
	# 현재 적용한 parameter 조합을 저장
	result['model']['model'+str(p)] = mlp_params_grid[p]
	# f1_score를 저장할 리스트 생성
	result['f1']['model'+str(p)] = []

	# mlp_params_grid[p] 딕셔너리를 인자로 넘겨줄 때 ** 사용
	# **을 사용하면 딕셔너리의 키-값 쌍을 풀어서 인자로 전달 가능
	# ex. MLPClassifier(random_state=seed, activation='relu', alpha=0.0001, hidden_layer_sizes=50, learning_rate_init=0.001, max_iter=50, solver='adam')
	model = MLPClassifier(random_state = seed, **mlp_params_grid[p])

	# Stratified K-fold 객체 생성
	skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)

	# validation f1_score 저장할 리스트 생성
	val_f1 = []

	# k번 만큼 실험을 반복하며, 매 회차마다의 train 데이터를 다시 train과 validation data 범위의 index를 각각 train_index와 val_index에 저장
	for train_idx, val_idx in skf.split(X, y):
		# train data와 validation data 생성
		X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
		X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

		# 모델에 학습
		model.fit(X_train, y_train)

		# X_train 데이터로 예측한 결과 저장
		train_predict = model.predict(X_train)

		# X_val 데이터로 예측한 결과 저장
		val_predict = model.predict(X_val)
		
		# val_predict의 f1_score를 하나씩 저장
		val_f1.append(f1_score(y_val, val_predict))

	# 각 모델 별 validation f1_score의 평균을 저장
	result['f1']['model'+str(p)].append(np.mean(val_f1))

# 결과 출력
result

# best parameter 찾기

# 모델의 f1_score로만 list 생성
f1_list = list(map(lambda x: np.mean(x[1]), result['f1'].items()))
# f1_score가 최대인 곳의 index 저장
f1_max_idx = f1_list.index(max(f1_list))

# best parameter 저장
best_param = result['model'][f'model{f1_max_idx}']

# best_param이 위치한 index 찾기
m = list(result['model'].keys())[list(result['model'].values()).index(best_param)]

# val result 출력

# best parameter인 곳의 validation f1_score 저장
f1 = result['f1'][m]

# best parameter와 validation의 f1_score 출력
print(f"best param: {best_param} \
		\n[validation result] \
		\nf1: {f1[0]:.3f}")

# test result 출력

# 최종으로 나온 best parameter의 MLP 모델
model = MLPClassifier(random_state = seed, **best_param)

# Stratified K-fold 객체 생성
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)

# k번 만큼 실험을 반복하며, 매 회차마다의 train 데이터를 다시 train과 validation data 범위의 index를 각각 train_index와 val_index에 저장
for train_idx, val_idx in skf.split(X, y):
	# train data와 validation data 생성
	X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
	X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

	# tox21_train 데이터로 학습
	model.fit(X_train, y_train)

# tox21_test 데이터로 예측
pred = model.predict(X_test)
pred_score = model.predict_proba(X_test)[:, 1]

# best parameter와 예측 결과의 성능 지표 출력
print(f'[test result] \
		\nbest param: {best_param} \
		\nprecision: {precision_score(y_test, pred):.3f} \
		\nrecall: {recall_score(y_test, pred):.3f} \
		\naccuracy: {accuracy_score(y_test, pred):.3f} \
		\nauc: {roc_auc_score(y_test, pred_score):.3f} \
		\nf1: {f1_score(y_test, pred):.3f}')
