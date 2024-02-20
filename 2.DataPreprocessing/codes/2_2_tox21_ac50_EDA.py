# 라이브러리 불러오기
    #1. pandas: 데이터를 빠르고 강력하게, 그리고 유연하고 쉽게 다룰 수 있게 해주는 데이터 분석, 증강 도구입니다.
    #    - <https://pandas.pydata.org/docs/user_guide/index.html>
import pandas as pd
    #2. matplolib: 정적, 애니메이션, 또는 상호작용형 시각화를 생성하기 위한 포괄적인 도구입니다.
    #    - <https://matplotlib.org/stable/users/index>
import matplotlib.pyplot as plt

# pandas 라이브러리를 통해 csv 파일 로드
folder = "."
data_file = 'tox21_ac50.csv'

df = pd.read_csv(f'{folder}/{data_file}')


# 전체 데이터 수 파악
print('전체 데이터 갯수:', len(df.index))
print()

# 데이터의 일부분 확인하기
print(df.head())
print()
# info() : columns에 대한 정보를 보여줌
# Null과 Dtype을 확인할 수 있음(columns이 많을 경우, 확인 불가)
print(df.iloc[:,-5:].info())

# AC50이 구간별로 얼마나 존재하는지 확인하는 그림
    # hist 함수를 통해 각 구간별로 데이터가 얼마나 존재하는지 확인
    # bins : 구간의 갯수
plt.hist(df['AC50'], bins= 10)
plt.title('AC50')
plt.show()

# 데이터 분포 확인하기
# botplot : 각 feature의 분포를 확인해주는 그림 - 최솟값, 최댓값, 4분위수, 이상치를 확인할 수 있음
plt.boxplot(df.iloc[:,-1])
plt.title('NR-AR Boxplot')
plt.show()
print()

# scatter : 좌표상에 점을 표시하여 변수간의 관계를 시각화해줌
    # alpha : 투명한 정도
    # c : 색 (AC50이 높을수록, 붉은색이고 낮을수록 파랑색을 갖도록 함) - 0~1의 값 입력
    # cmap : 나타나는 색의 테마를 설정
plt.scatter(df.iloc[:,42], df.iloc[:,46], alpha= 0.1, c= df.iloc[:,-1]/100, cmap= 'bwr')    # c의 값이 0~1의 값을 갖도록 100을 나눠줌
plt.xlabel('maccs_42')  # x축의 이름
plt.ylabel('maccs_46')  # y축의 이름
plt.colorbar()      # colorbar를 표시 - 각각의 색이 어느정도의 값을 나타내는지 표시
plt.show()

