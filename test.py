import requests

URL = 'http://127.0.0.1:5000/'

if __name__ == '__main__':
	sentence = "hi"
	response = requests.get(URL+"chat/"+sentence)
	data = response.json()

	if 'error' in data:
		print(f'ERROR: {data["error"]}')
	else:
		print(f'{data["data"]}')