# LINE 과제 결과 Report



- model_trainer.py, server.py, preprocess.py, eda_process.py, EDA.ipynb 파일을 모두 같은 폴더에 넣고 시작.
- 해당 폴더 하위에 'model' 이라는 이름으로 폴더 만들고 시작.
- 'model' 폴더에 'model.dat' 데이터 삽입.
- 해당 폴더 하위에 'data' 라는 폴더 만들고 시작.
- 'data' 폴더에 criteo data 압축 풀고 시작.

## 예측기 실행방법

- python 3.7 환경에서 model_trainer.ipynb 파일 전체 실행

## Web Service 실행방법

- 반드시 예측기 실행 후 실행.
- python 3.7 환경에서 server.py 파일 실행.
- http://127.0.0.1:5000/update/model 로 이동
- put line parameter에 주어진 데이터 포맷에 맞추어 데이터 입력 후, '제출' 버튼 클릭
- json type 의 데이터에서 'cvr' 파라미터에 해당 'liine' 데이터의 예측 cvr 값이 출력됨. 

## 분석결과

- 평가지표 : AUC
- 학습 데이터 : 0.6713
- 평가 데이터 : 0.6617

## 피쳐 선택 시 , price_product 는 Sale이 1일 때 항상 0보다 크므로, 정답 Label 인 Sale 과 종속성이 너무 크기에 제외하고 CVR을 예측.

- price_product를 제외하고 데이터를 가공하여 사용 할 경우, 다른 데이터들과 CVR을 구하는 과정을 더욱 정확히 확인 할 수 있음.
- price_product를 넣어서 성능을 증가시켜야 하는 상황에는 , 위의 integer_columns 에 price_product 를 추가하는 것으로 성능을 향상 시킬 수 있음.



## EDA

- ### 추가 분석결과  : EDA.ipynb 파일에 분석 코드 및 결과 저장

- product_price 분석

  - 총 150만개의 데이터 중 7890개 데이터를 제외하면
    product_price 는 Sale 이 1인 경우 항상 0 이상 , 아닌경우 항상 0이다.
  - 두 데이터의 종속성이 너무 높은 것으로 판단됨

- Sale 비율 : 약 0.1

- click_timestamp 를 이용하여 weekday,hour 등의 데이터로 분리 후 분석

  - 요일별 Sale 비율 데이터 분석결과 : 토,일,월요일의 데이터가 비교적 값이 큼
  - 시간별 비율 데이터 분석 결과 : 22~02시 까지의 데이터가 비교적 값이 큼

- Group 별 시각화

  - country , weekday 그룹별 데이터 시각화 결과 Sale 비율 값이 국가,요일별로 다르게 나타나는 것으로 확인
  - country, hour 그룹별 데이터 시각화 결과 Sale 비율 값이 국가,요일별로 다르게 나타나는 것으로 확인

  

## Model

- Logistic regression
- L1 regularization Customize로 구현하여 사용. Lambda = 0.001
- optimizer : adam

## Features

- User feature , ad feature , time feature를 각각 만든 후,

- Users X ad Feature , ad teature X time feature 로 Second term 을 생성.

- X 는 cross term 생성을 의미

- product_country X Time Feature 로 cross term 피쳐 생성.

- device_type X  product_gender 로 cross term 피쳐 생성

- User Feature

  - device_type

- Ad Feature

  - nb_clicks_1week
  - product_gender
  - product_age_group
  - product_country

- Time Feature : click_timestamp 를 이용하여 생성

  - weekday
  - hour

- Cross_term

  - product_country_weekday
  - product_country_hour
  - device_type_product_gender

  

## Data Preprocessing

- 타입에 따른 데이터 분류 : 
  - 범주형 데이터 :device_type,product_gender,product_age_group,product_country
  - 정수형 데이터 :nb_clicks_1week

- 결측값 처리
  - 범주형 데이터 : 결측값을 제외하고 최빈값이 전체 데이터의 50% 가 넘으면 결측값을 최빈값으로 변경
  - 정수형 데이터 : 결측값을 전체 데이터의 평균으로 대체



## 범주형 데이터 - 학습 데이터에는 없고, 훈련데이터에만 포함된 데이터에 대한 처리

- - 학습 데이터에 100만개에는 해당 값이 없고, 훈련데이터에만 포함된 데이터에 대한 처리.

  - One - hot - encoding 과정을 진행 할 때, 미리 작업하여 저장해둔 column들로 reindex 하면,

    새로 등장한 데이터는 자연스럽게 column에서 사라지게 됨..



## version1 - 기초 데이터 사용 및 가공 및 결과

Ver1. 범주형 데이터 one hot encoding , normalization, 결측치 처리  

- 10만개 데이터를 2:1로 잘라서 train / valid 데이터로 나누어서 테스트

- 1 epochs, batch_size=1 , optimizer : Adam 사용.
- tensorflow 로 단일 레이어 구성하여 logistic regression 으로 테스트

| ver  |         | 내용                                                         | Loss                                 | 평가지표        |
| ---- | ------- | ------------------------------------------------------------ | ------------------------------------ | --------------- |
| ver1 | ver 1.1 | nb_clicks_1week 만 사용                                      | val_binary_crossentropy: 0.3390      | val_auc: 0.5398 |
|      | ver 1.2 | 범주형 데이터 추가('device_type','product_gender'<br />,'product_age_group','product_country') | val_binary_crossentropy: 0.3964      | val_auc: 0.6351 |
|      | ver 1.3 | nb_clicks_1week 피쳐에 대해, std_norm 진행                   | val_binary_crossentropy: 0.3192      | val_auc: 0.6736 |
|      | ver 1.4 | nb_clicks_1week 피쳐에 대해, std_norm 이후 minmax_norm 진행  | val_binary_crossentropy: 0.3196      | val_auc: 0.6722 |
|      | ver 1.5 | nb_clicks_1week 피쳐 -1 값을 0으로 대체                      | \- val_binary_crossentropy: 0.3203 - | val_auc: 0.6702 |
|      | ver 1.6 | nb_clicks_1week 피쳐 -1 값을 평균으로 대체                   | val_binary_crossentropy: 0.3198 -    | val_auc: 0.6707 |
|      | ver 1.7 | 범주형 데이터들 중, -1을 모두 제외한 후,<br />최빈값이 전체의 50%가 넘을 경우 결측치를 최빈값으로 대체 | val_binary_crossentropy: 0.3198 -    | val_auc: 0.6714 |



## version2 - 제출 format에 맞춘 data로 테스트 및 cross Term 추가



Ver. 2. 시간 데이터 추가,  cross_Term 추가 , 100만개 데이터로 학습, 50만개 데이터로 평가

Batch_size =. 100 , epochs=5

| ver  |         | 내용                              | 학습데이터 결과 | 평가 데이터 결과 |
| ---- | ------- | --------------------------------- | --------------- | ---------------- |
| Ver2 | ver 2.1 | Product_country X time 추가       | auc: 0.6711     | auc: 0.6636      |
|      | ver 2.2 | Product_gender X device_type 추가 | auc: 0.6713     | auc: 0.6617      |
|      | ver 2.3 | L1 정규화 추가                    | auc: 0.6625     | auc: 0.6600      |

