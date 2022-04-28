import logging
import time
import os
import sys
import random
import json

sys.path.insert(1, '/src/DNNC-few-shot-intent')
sys.path.insert(1, '/src/DNNC-few-shot-intent/models')

from flask import Flask, request, jsonify
from healthcheck import HealthCheck
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from models.utils import InputExample
from models.dnnc import DNNC
from models.dnnc import ENTAILMENT, NON_ENTAILMENT
from intent_predictor import DnncIntentPredictor


class MockArgs:
    def __init__(self):
        self.seed = 42
        self.bert_model = 'roberta-base'
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.learning_rate = 1e-5
        self.num_train_epochs = 7
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.no_cuda = False
        self.gradient_accumulation_steps = 2
        self.max_grad_norm = 1.0
        self.label_smoothing = 0.1
        self.max_seq_length = 128
        self.do_lower_case = False
        self.bert_nli_path = "/src/DNNC-few-shot-intent/roberta_nli"
        self.train_file_path = "/src/example.json"
        self.save_model_path = ''
        self.output_dir = None


def train_model():
    args = MockArgs()

    random.seed(args.seed)

    with open(args.train_file_path) as f:
        train_data = json.load(f)
    print(train_data)


    nli_train_examples = []
    nli_dev_examples = []

    tasks = [{'task': i, 'examples': exs} for i, exs in train_data.items()]
    all_entailment_examples = []
    all_non_entailment_examples = []

    # entailement
    for task in tasks:
        examples = task['examples']
        for j in range(len(examples)):
            for k in range(len(examples)):
                if k <= j:
                    continue

                all_entailment_examples.append(InputExample(examples[j], examples[k], ENTAILMENT))
                all_entailment_examples.append(InputExample(examples[k], examples[j], ENTAILMENT))

    # non entailment
    for task_1 in range(len(tasks)):
        for task_2 in range(len(tasks)):
            if task_2 <= task_1:
                continue
            examples_1 = tasks[task_1]['examples']
            examples_2 = tasks[task_2]['examples']
            for j in range(len(examples_1)):
                for k in range(len(examples_2)):
                    all_non_entailment_examples.append(InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
                    all_non_entailment_examples.append(InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))                    

    nli_train_examples = all_entailment_examples + all_non_entailment_examples
    nli_dev_examples = all_entailment_examples[:100] + all_non_entailment_examples[:100] # sanity check for over-fitting

    model = DNNC(path = args.bert_nli_path, args = args)

    model.train(nli_train_examples, nli_dev_examples)

    intent_predictor = DnncIntentPredictor(model, tasks)

    print(intent_predictor.predict_intent("Hi there"))
    return intent_predictor

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), integrations=[FlaskIntegration()])


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# PRETRAINED_MODEL_NAME_OR_PATH = os.environ.get("PRETRAINED_MODEL_NAME_OR_PATH")
# logging.info(f"PRETRAINED_MODEL_NAME_OR_PATH = {PRETRAINED_MODEL_NAME_OR_PATH}")
# MASK_ID = 103
# try:
#     cuda = torch.cuda.is_available()
#     if cuda:
#         torch.cuda.set_device(0)  # singe gpu
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     logger.info(f"masked_lm is set to run on {device}")

#     # init model
#     tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
#     model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
#     model.eval()
#     if cuda:
#         model.cuda()

#     logger.info("masked_lm model is ready")
# except Exception as e:
#     sentry_sdk.capture_exception(e)
#     logger.exception(e)
#     raise e

intent_predictor = train_model()
app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")
logging.getLogger("werkzeug").setLevel("WARNING")


@app.route("/respond", methods=["POST"])
def respond():
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

# test with 
# curl -X POST 0.0.0.0:1234/respond -H 'Content-Type: application/json' -d '{"text": "Hi there"}'
