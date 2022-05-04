from model import InputExample, prepare_examples, \
ENTAILMENT, NON_ENTAILMENT, MockArgs, DNNC, train_model
import pytest
from unittest.mock import patch
import datetime
import json
import dataclasses
import typing as tp

@dataclasses.dataclass
class PrepareCase:
    test_case: tp.List[tp.Dict[str, tp.Any]]
    ground_truth: tp.List[InputExample]

def custom_ie_compare(self, other):
    return self.text_a == other.text_a and\
        self.text_b  == other.text_b and\
        self.label == other.label


PREPARE_EXAMPLES_CASES = [
    PrepareCase(
        test_case = [
            {
                'task': "greet",
                'examples': ['hi', 'hello'],
            },
            {
                'task': "bye",
                'examples': ['bye', 'goodbye'],
            },
        ],
        ground_truth = [
            InputExample('hi', 'hello', ENTAILMENT),
            InputExample('hello', 'hi', ENTAILMENT),
            InputExample('bye', 'goodbye', ENTAILMENT),
            InputExample('goodbye', 'bye', ENTAILMENT),
            InputExample('hello', 'bye', NON_ENTAILMENT),
            InputExample('hello', 'goodbye', NON_ENTAILMENT),
            InputExample('hi', 'bye', NON_ENTAILMENT),
            InputExample('hi', 'goodbye', NON_ENTAILMENT),
            InputExample('bye', 'hello', NON_ENTAILMENT),
            InputExample('goodbye', 'hello', NON_ENTAILMENT),
            InputExample('bye', 'hi', NON_ENTAILMENT),
            InputExample('goodbye', 'hi', NON_ENTAILMENT),
        ]
    ),
    PrepareCase(
        test_case = [
            {
                'task': "greet",
                'examples': ['hello'],
            },
            {
                'task': "bye",
                'examples': ['bye', 'goodbye'],
            },
        ],
        ground_truth = [
            InputExample('bye', 'goodbye', ENTAILMENT),
            InputExample('goodbye', 'bye', ENTAILMENT),
            InputExample('hello', 'bye', NON_ENTAILMENT),
            InputExample('hello', 'goodbye', NON_ENTAILMENT),
            InputExample('bye', 'hello', NON_ENTAILMENT),
            InputExample('goodbye', 'hello', NON_ENTAILMENT),
        ]
    ),
]

@pytest.mark.parametrize('case', PREPARE_EXAMPLES_CASES)
@patch("models.utils.InputExample.__eq__", custom_ie_compare)
def test_prepare_examples(case: PrepareCase):
    test_case = case.test_case

    ground_truth = case.ground_truth

    key = lambda x: (x.text_a, x.text_b, x.label)

    train_res, dev_res = prepare_examples(test_case)
    assert train_res == dev_res
    assert InputExample('hi', 'hello', ENTAILMENT) == InputExample('hi', 'hello', ENTAILMENT)
    assert sorted(ground_truth, key=key) == sorted(train_res, key=key)


def test_model():
    test_case = {
        "greet": [
            "Howdy friend!",
            "Hello dear",
            "Oh hi pal!",
            "Hello everyone",
            "Bonjour mon ami",
            "Nice to see you again",
            "Good evening!"
        ],
        "bye": [
            "Bye bye sweetie!",
            "See you later",
            "Goodbye friend!",
            "See you soon!",
            "Alright then"
        ]
    }

    args = MockArgs()
    args.num_train_epochs = 1

    tasks = [{'task': i, 'examples': exs} for i, exs in test_case.items()]
    nli_train_examples, nli_dev_examples = prepare_examples(tasks)

    model = DNNC(path = args.bert_nli_path, args = args)

    pre_score = model.evaluate(nli_train_examples)

    start = datetime.datetime.now()
    model.train(nli_train_examples, nli_dev_examples)
    train_time = (datetime.datetime.now() - start).total_seconds()

    after_score = model.evaluate(nli_train_examples)

    assert (after_score - pre_score) > 0.01 
    assert train_time < 5 * 60

def test_train_model():
    args = MockArgs()

    start = datetime.datetime.now()
    predictor = train_model(args)
    train_time = (datetime.datetime.now() - start).total_seconds()

    with open(args.train_file_path) as f:
        train_data = json.load(f)

    for k in train_data:
        for u in train_data[k]:
            start = datetime.datetime.now()
            i, sim, sim_u = predictor.predict_intent(u)
            eval_time = (datetime.datetime.now() - start).total_seconds()
            assert eval_time < 0.5
            assert i == k
            assert sim > 0.8

    i, sim, sim_u = predictor.predict_intent('Hello there')
    assert i == 'greet'

    assert train_time < 15 * 60
