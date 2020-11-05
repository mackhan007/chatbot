from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import random
import chatbot

app = Flask(__name__)
api = Api(app)

chatbot = chatbot.chatbot()

class chat(Resource):
	def get(self, sentance):
		try:
			reply = chatbot.getresponse(sentance)
			return {"data": reply}
		except:
			abort(404, message="Video doesn't exist, cannot update")


api.add_resource(chat, "/chat/<string:sentance>")

if __name__ == '__main__':
	app.run(debug=True)