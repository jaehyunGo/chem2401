# 라이브러리 불러오기
    #1. pandas: 데이터를 빠르고 강력하게, 그리고 유연하고 쉽게 다룰 수 있게 해주는 데이터 분석, 증강 도구입니다.
    #    - <https://pandas.pydata.org/docs/user_guide/index.html>
import pandas as pd
    #2. matplolib: 정적, 애니메이션, 또는 상호작용형 시각화를 생성하기 위한 포괄적인 도구입니다.
    #    - <https://matplotlib.org/stable/users/index>
import matplotlib.pyplot as plt

# pandas 라이브러리를 통해 csv 파일 로드
folder = "."
data_file = 'tox21_dataset.csv'

df = pd.read_csv(f'{folder}/{data_file}')

# 전체 데이터 수 파악
print('전체 데이터 갯수:', len(df.index))

# 데이터의 일부분 확인하기
print(df.head())

# label 분포확인
index_label = -1    # NR_AR label의 index

# 각 label에 해당하는 데이터의 수 구하기
    # loc:  인덱스(index)나 컬럼(column)의 이름을 기준으로 행과 열을 선택합니다.
    # iloc: 위치(인덱스나 컬럼의 숫자)를 기준으로 행과 열을 선택합니다.
label_0 = len(df.loc[df.iloc[:,index_label] == 0])  # label이 0인 데이터
label_1 = len(df.loc[df.iloc[:,index_label] == 1])  # label이 1인 데이터
label_nan = 7831 - (label_0 + label_1)

print('NR-AR의 각 class별 갯수')
print('  - 0인 데이터의 수:',label_0)
print('  - 1인 데이터의 수:',label_1)
print('  - 결측값인 데이터의 수:',label_nan)
print()

# bar 그래프를 통해 data imbalance를 시각화
plt.bar([0,1,2], [label_0, label_1, label_nan])
plt.xticks([0,1,2], [0,1,'NaN'])
plt.title(f'{df.columns[index_label]}')
plt.xlabel('label')
plt.ylabel('Num of Data')
plt.show()

# feature 분포 확인
index_label = 42    # 1(maccs_1) ~ 166(maccs_166) 중 선택

# 각 label에 해당하는 데이터의 수 구하기
label_0 = len(df.loc[df.iloc[:,index_label] == 0])
label_1 = len(df.loc[df.iloc[:,index_label] == 1])

print(f'maccs_{index_label}의 분포')
print('  - 0인 데이터의 수:',label_0)
print('  - 1인 데이터의 수:',label_1)
print()

# bar 그래프를 통해 data imbalance를 시각화
plt.bar([0,1], [label_0, label_1])
plt.xticks([0,1], [0,1])
plt.title(f'{df.columns[index_label]}')
plt.xlabel('label')
plt.ylabel('Num of Data')
plt.show()