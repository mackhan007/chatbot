from flask import Flask, redirect
from flask_restful import Api, Resource
import chatbot1

app = Flask(__name__)
api = Api(app)

chatbot = chatbot1.chatbot()
chatbot.load_model()


class chat(Resource):
    def get(self, sentance):
        try:
            reply = chatbot.getresponse(sentance)
            return {"data": reply}
        except:
            return {"data": "unable to get response"}


class Main(Resource):
    def get(self):
        return "Welcome to the chatbot, goto /chat/<sentance> to get a response"


api.add_resource(chat, "/chat/<string:sentance>")
api.add_resource(Main, "/")

if __name__ == '__main__':
    app.run(debug=True)
