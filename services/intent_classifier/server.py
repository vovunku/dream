import logging
import time
import os

from flask import Flask, request, jsonify
from healthcheck import HealthCheck
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from model import train_model, MockArgs
from flask import current_app, Blueprint, render_template

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), integrations=[FlaskIntegration()])


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

intent_predictor = train_model(MockArgs())


respond = Blueprint('respond', __name__, url_prefix='/')
@respond.route("/respond", methods=["POST"])
def respond_fun():
    st_time = time.time()

    text = request.json.get("text", [])
    try:
        intent_prediction = intent_predictor.predict_intent(text)
    except Exception as exc:
        logger.exception(exc)
        sentry_sdk.capture_exception(exc)
        intent_prediction = None

    total_time = time.time() - st_time
    logger.info(f"DNNC exec time: {total_time:.3f}s")
    return jsonify({"predicted": intent_prediction})


def create_app():
    app = Flask(__name__)

    app.register_blueprint(respond)

    return app

app = create_app()

health = HealthCheck(app, "/healthcheck")
logging.getLogger("werkzeug").setLevel("WARNING")

# test with 
# curl -X POST 0.0.0.0:1234/respond -H 'Content-Type: application/json' -d '{"text": "Hi there"}'
