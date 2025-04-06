from flask import Flask, request
from alexa_announcer import speak_for_state

app = Flask(__name__)

@app.route("/trigger", methods=["POST"])
def trigger():
    state = request.json.get("state")
    if state:
        speak_for_state(state)
        return {"status": "spoken"}
    return {"status": "no state received"}, 400

if __name__ == "__main__":
    app.run(port=5000)
