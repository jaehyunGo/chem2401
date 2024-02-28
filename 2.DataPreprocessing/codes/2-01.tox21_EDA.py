# 전처리전 데이터 EDA
    # 데이터를 불러오거나 가공하는데 활용하는 라이브러리
import pandas as pd
    # 데이터 그래프 혹은 도표를 시각화해주는 라이브러리
import matplotlib.pyplot as plt
# 파일 경로 설정
folder = '.'
file = 'tox21.xlsm' # 12개의 label과 SMILE가 존재하는 데이터 파일
cur_sheet = 'Tox21'

# pandas 라이브러리를 통해 csv파일을 불러오기
df = pd.read_excel(f'{folder}/{file}', sheet_name= cur_sheet)

# 데이터 프레임 출력
print()
print('='*100)
print('데이터 프레임 출력')
print(df)

# 데이터프레임 칼럼명 출력
print()
print('='*100)
print('데이터프레임 칼럼명 출력')
print(df.columns)

# 데이터프레임 정보 출력(칼럼명, null값, Dtype확인 가능)
print()
print('='*100)
print('데이터프레임 정보 출력(칼럼명, null값, Dtype확인 가능)')
print(df.info())

# 데이터 프레임 처음부터 5개의 행 출력
print()
print('='*100)
print('데이터 프레임 처음부터 5개의 행 출력')
print(df.head())

# 데이터 프레임 끝에서부터 10개의 행 출력
print()
print('='*100)
print('데이터 프레임 끝에서부터 10개의 행 출력')
print(df.tail(10))

# null값 개수 출력
print()
print('='*100)
print('null값 개수 출력')
missing_values_count = df.isna().sum()
print(missing_values_count)

# 결측값 확인
fig, axes = plt.subplots(2,6, figsize= (15,6))

for label_idx in range(12):
    label_0 = len(df.loc[df.iloc[:, label_idx] == 0])  # label이 0인 데이터
    label_1 = len(df.loc[df.iloc[:, label_idx] == 1])  # label이 1인 데이터
    label_nan = len(df[df.iloc[:, label_idx].isna()])
    axes[label_idx//6, label_idx%6].bar([0,1,2], [label_0, label_1, label_nan])
    axes[label_idx//6, label_idx%6].set_xticks([0,1,2], [0,1,'NaN'])
    axes[label_idx//6, label_idx%6].set_title(f'{df.columns[label_idx]}')
    axes[label_idx//6, label_idx%6].set_xlabel('label')
    axes[label_idx//6, label_idx%6].set_ylabel('Num of Data')
plt.tight_layout()
plt.show()

