import pandas as pd
cars = pd.read_csv("[DKU]DeepLearning_Cloud\cars.csv")



# (1) 데이터셋의 위쪽 5행을 보이시오
cars.head(5)

# (2) 데이터셋의 컬럼들 이름을 보이시오
cars.columns

# (3) 데이터셋의 두 번째 컬럼의 값들만 보이시오.
cars['dist']

# (4) 데이터셋의 11~20행 자료중 speed 컬럼의 값들만 보이시오.
cars['speed'][11:21]

# (5) speed 가 20 이상인 행들의 자료만 보이시오
cars[cars['speed']>=20]

# (6) speed 가 10 보다 크고 dist 가 50보다 큰 행들의 자료만 보이시오.
cars[(cars['speed']>20) & (cars['dist'] > 50)]

# (7) speed 가 15 보다 크고 dist 가 50보다 큰 행들은 몇 개인지 보이시오
sp = cars['speed'] > 15
dis = cars['dist'] > 50
cars[sp & dis].count()