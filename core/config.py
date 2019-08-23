from state_formatters.dp_formatters import *

TELEGRAM_TOKEN = ''
TELEGRAM_PROXY = ''

DB_NAME = 'test'
HOST = '127.0.0.1'
PORT = 27017

MAX_WORKERS = 1

SKILLS = [
    {
        "name": "alice",
        "protocol": "http",
        "host": "127.0.0.1",
        "port": 8000,
        "endpoint": "respond",
        "external": True,
        "path": "",
        "formatter": alice_formatter
    },
    {
        "name": "aiml",
        "protocol": "http",
        "host": "127.0.0.1",
        "port": 2080,
        "endpoint": "aiml",
        "path": "skills/aiml/aiml_skill.json",
        "env": {
            "CUDA_VISIBLE_DEVICES": ""
        },
        "gpu": False,
        "formatter": aiml_formatter
    }
]

ANNOTATORS = [
]

SKILL_SELECTORS = [
    {
        "name": "rule_based_selector",
        "protocol": "http",
        "host": "127.0.0.1",
        "port": 2083,
        "endpoint": "skill_names",
        "path": "skill_selectors/alexa_skill_selectors/rule_based_selector.json",
        "env": {
            "CUDA_VISIBLE_DEVICES": ""
        },
        "gpu": False,
        "formatter": base_skill_selector_formatter
    }
]
RESPONSE_SELECTORS = [
]

POSTPROCESSORS = []
