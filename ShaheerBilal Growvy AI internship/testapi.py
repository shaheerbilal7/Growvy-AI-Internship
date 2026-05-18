import requests

# The URL of your running Flask server
url = 'http://127.0.0.1:5000/predict'

# Path to an image to classify (change 'test_image.jpg' to a real filename)
files = {'image': open('test_image.jpg', 'rb')}

response = requests.post(url, files=files)
print(response.json())