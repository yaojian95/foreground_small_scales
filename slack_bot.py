import slack
from flask import Flask, request, Response
from slackeventsapi import SlackEventAdapter
import datetime

app = Flask(__name__)
signing_secrete = '8399c5f959906a3e5c97d3e50e5dfc21'
slack_event_adapter = SlackEventAdapter(signing_secrete, '/slack/events',app)
client = slack.WebClient(token='xoxb-2853441490869-3270335584390-2DTxCSLwmv3OB0lEdPB8Sw2a')
BOT_ID = client.api_call("auth.test")['user_id']

epoch = 0

@slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')

    text = event.get('text')
    if BOT_ID!=user_id:

        client.chat_postMessage(channel=channel_id, text=f'Now is at epoch: {epoch}')

# @app.route('/message-count', methods=['POST'])
# def message_count():
#     data = request.form
#     user_id = data.get('user_id')
#     channel_id = data.get('channel_id')
#     message_count = message_counts.get(user_id, 0)
#     client.chat_postMessage(channel=channel_id, text=f"got: {message_count}")
#     return Response(), 200

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()