import os
import flask
from flask import request, Flask
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)


@app.route('/answer', methods=['POST'])
def answer():
    global tokenizer, model

    param = request.json
    sentence = param['sentence']

    response, _ = model.chat(tokenizer, sentence, history=[])

    result = {
        'ans': response
    }

    print("human ", sentence, "chatglm: ", response)
    return result


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(os.path.join('..', 'models', 'chatglm-6b'), trust_remote_code=True)
    model = AutoModel.from_pretrained(os.path.join('..', 'models', 'chatglm-6b'), trust_remote_code=True).half().cuda()
    model = model.eval()

    app.run(host='0.0.0.0', port=8088)