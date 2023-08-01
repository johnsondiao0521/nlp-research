import requests


if __name__ == '__main__':
    url = 'http://192.168.1.73:8088/answer'
    while True:
        text = input("我:")
        param = {
            "sentence": text
        }

        res = requests.post(url, json=param)
        res = res.json()

        ans = res['ans']

        print("chatglm: ", ans)