import requests
# 이미지 요청 패키지
from PIL import Image
from io import BytesIO

# test
# response = requests.get('http://www.naver.com')
# 200 제대로 접속이 됐다.
# 400 잘못된 주소
# 500 서버 에러

vision_base_url = 'https://danuser6computervision.cognitiveservices.azure.com/vision/v2.0/'

vision_base_url += 'analyze'

image_url = 'https://www.dankook.ac.kr/html_portlet_repositories/images/ExtImgFile/10158/10185/343623/59734.jpg'

image = Image.open(BytesIO(requests.get(image_url).content))

headers = {'Ocp-Apim-Subscription-Key': subscription_key}
params = {'visualFeatures': 'Categories,Description,Color'}
data = {'url': image_url}

response = requests.post(vision_base_url, json=data, params=params, headers=headers)

result = response.json()

print(result)
