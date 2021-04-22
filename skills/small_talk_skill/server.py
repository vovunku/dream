#!/usr/bin/env python

import logging
import time
import re
from random import choice
import json

from flask import Flask, request, jsonify
from os import getenv
import sentry_sdk

from common.constants import CAN_NOT_CONTINUE, CAN_CONTINUE_SCENARIO
from common.utils import get_skill_outputs_from_dialog, get_sentiment, get_entities
from common.universal_templates import if_choose_topic, if_switch_topic, if_chat_about_particular_topic


sentry_sdk.init(getenv('SENTRY_DSN'))

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


with open("topic_words.json", "r") as f:
    TOPIC_PATTERNS = json.load(f)

for topic in TOPIC_PATTERNS:
    words = TOPIC_PATTERNS[topic]
    pattern = "(" + "|".join([r'\b%s' % word for word in words]) + ")"
    TOPIC_PATTERNS[topic] = re.compile(pattern, re.IGNORECASE)

with open("small_talk_scripts.json", "r") as f:
    TOPIC_SCRIPTS = json.load(f)

USER_TOPIC_START_CONFIDENCE = 0.98
FOUND_WORD_START_CONFIDENCE = 0.5
BOT_TOPIC_START_CONFIDENCE = 0.7
CONTINUE_CONFIDENCE = 0.99
LONG_ANSWER_CONTINUE_CONFIDENCE = 1.0
YES_CONTINUE_CONFIDENCE = 1.0
# if let's chat about TOPIC [key-words]
NOT_SCRIPTED_TOPICS = [
    'cars', "depression", "family", "life", "love", "me", "politics", "science",
    # TODO: remove science when dff-science-skill will be merged
    "school", "sex", "star wars", "donald trump", "work", "you"
]


@app.route("/respond", methods=['POST'])
def respond():
    st_time = time.time()
    dialogs_batch = request.json["dialogs"]
    confidences = []
    responses = []
    human_attributes = []
    bot_attributes = []
    attributes = []

    for dialog in dialogs_batch:
        human_attr = dialog["human"]["attributes"]
        used_topics = human_attr.get("small_talk_topics", [])
        human_attr = {}
        bot_attr = {}
        attr = {}

        skill_outputs = get_skill_outputs_from_dialog(
            dialog["utterances"][-3:], skill_name="small_talk_skill", activated=True)
        if len(skill_outputs) > 0:
            # small_talk_skill was active on the previous step
            topic = skill_outputs[0].get("small_talk_topic", "")
            script_step = skill_outputs[0].get("small_talk_step", 0)
            script = skill_outputs[0].get("small_talk_script", [])
            logger.info(f"Found previous step topic: `{topic}`.")
        else:
            topic = ""
            script_step = 0
            script = []

        _, new_user_topic, new_conf = pickup_topic_and_start_small_talk(dialog)
        logger.info(f"From current user utterance: `{dialog['human_utterances'][-1]['text']}` "
                    f"extracted topic: `{new_user_topic}`.")
        sentiment = get_sentiment(dialog["human_utterances"][-1], probs=False)[0]
        if len(topic) > 0 and len(script) > 0 and \
                (len(new_user_topic) == 0 or new_conf == FOUND_WORD_START_CONFIDENCE):
            # we continue dialog if new topic was not found or was found just as the key word in user sentence.
            # because we can start a conversation picking up topic with key word with small proba
            if sentiment == "negative":
                logger.info("Found negative sentiment to small talk phrase. Finish script.")
                response, confidence, attr = "", 0.0, {"can_continue": CAN_NOT_CONTINUE,
                                                       "small_talk_topic": "", "small_talk_step": 0,
                                                       "small_talk_script": []}
            else:
                response, confidence, attr = get_next_response_on_topic(
                    topic, dialog["human_utterances"][-1], curr_step=script_step + 1, topic_script=script)
            if response != "":
                logger.info(f"Continue script on topic: `{topic}`.\n"
                            f"User utterance: `{dialog['human_utterances'][-1]['text']}`.\n"
                            f"Bot response: `{response}`.")
        else:
            logger.info("Try to extract topic from user utterance or offer if requested.")
            response, topic, confidence = pickup_topic_and_start_small_talk(dialog)
            if len(topic) > 0 and topic not in used_topics:
                logger.info(f"Starting script on topic: `{topic}`.\n"
                            f"User utterance: `{dialog['human_utterances'][-1]['text']}`.\n"
                            f"Bot response: `{response}`.")
                # topic script start, response is already formulated
                human_attr["small_talk_topics"] = used_topics + [topic]
                attr["response_parts"] = ["prompt"]
                attr["can_continue"] = CAN_CONTINUE_SCENARIO
                attr["small_talk_topic"] = topic
                attr["small_talk_step"] = 0
                attr["small_talk_script"] = TOPIC_SCRIPTS.get(topic, [])
            else:
                logger.info(f"Can not extract or offer topic.")

        if len(response) == 0:
            confidence = 0.

        responses.append(response)
        confidences.append(confidence)
        human_attributes.append(human_attr)
        bot_attributes.append(bot_attr)
        attributes.append(attr)

    total_time = time.time() - st_time
    logger.info(f'small_talk_skill exec time: {total_time:.3f}s')
    return jsonify(list(zip(responses, confidences, human_attributes, bot_attributes, attributes)))


def get_next_response_on_topic(topic, curr_user_uttr, curr_step=0, topic_script=None):
    topic_script = [] if topic_script is None else topic_script
    attr = {}

    if curr_step == len(topic_script):
        # prev_bot_uttr was the last in the script
        # can not continue with the same script&topic
        logger.info("Script was finished.")
        attr["can_continue"] = CAN_NOT_CONTINUE
        attr["small_talk_topic"] = ""
        attr["small_talk_step"] = 0
        attr["small_talk_script"] = []
        return "", 0.0, attr

    if isinstance(topic_script[curr_step], str):
        next_bot_uttr = topic_script[curr_step]
        attr["can_continue"] = CAN_CONTINUE_SCENARIO
        attr["small_talk_topic"] = topic
        attr["small_talk_step"] = curr_step
        if len(curr_user_uttr["text"].split()) > 7:
            confidence = LONG_ANSWER_CONTINUE_CONFIDENCE
        else:
            confidence = CONTINUE_CONFIDENCE
    elif isinstance(topic_script[curr_step], dict):
        yes_detected = curr_user_uttr["annotations"].get("intent_catcher", {}).get("yes", {}).get("detected", 0) == 1
        if yes_detected:
            next_bot_uttr = topic_script[curr_step]["yes"]
            attr["can_continue"] = CAN_CONTINUE_SCENARIO
            attr["small_talk_topic"] = topic
            attr["small_talk_step"] = curr_step
            confidence = YES_CONTINUE_CONFIDENCE
        else:
            # consider all other answers as NO
            next_bot_uttr = topic_script[curr_step]["no"]
            attr["can_continue"] = CAN_CONTINUE_SCENARIO
            attr["small_talk_topic"] = topic
            attr["small_talk_step"] = curr_step
            if len(curr_user_uttr["text"].split()) > 7:
                confidence = LONG_ANSWER_CONTINUE_CONFIDENCE
            else:
                confidence = CONTINUE_CONFIDENCE
    else:
        next_bot_uttr = ""
        confidence = 0.

    if isinstance(next_bot_uttr, list):
        if len(next_bot_uttr) == 0:
            logger.info("Script was finished.")
            attr["can_continue"] = CAN_NOT_CONTINUE
            attr["small_talk_topic"] = ""
            attr["small_talk_step"] = 0
            attr["small_talk_script"] = []
            return "", 0.0, attr
        attr["small_talk_script"] = topic_script[:curr_step] + next_bot_uttr + topic_script[curr_step + 1:]
        next_bot_uttr = attr["small_talk_script"][curr_step]
    else:
        attr["small_talk_script"] = topic_script[:curr_step] + [next_bot_uttr] + topic_script[curr_step + 1:]
    return next_bot_uttr, confidence, attr


def offer_topic(dialog):
    """
    There is an opportunity to choose topic taking into account the dialog history.
    For now, it's just random pick up from `TOPIC_WORDS.keys()`.

    Args:
        dialog: dialog from agent

    Returns:
        string topic out of `TOPIC_WORDS.keys()`
    """
    used_topics = dialog["human"]["attributes"].get("small_talk_topics", [])
    topic_set = set(TOPIC_PATTERNS.keys()).difference(set(used_topics)).difference(
        {"sex", "me", "politics", "depression", "donald trump", "news", "school", "star wars", "work", "you"})
    if len(topic_set) > 0:
        topic = choice(list(topic_set))
    else:
        topic = ""
    return topic


def find_topics_in_substring(substring):
    """
    Search topic words in the given string

    Args:
        substring: any string

    Returns:
        list of topics out of `TOPIC_WORDS.keys()`
    """
    topics = []
    for topic in TOPIC_PATTERNS:
        if re.search(TOPIC_PATTERNS.get(topic, "XXXXX"), substring):
            topics.append(topic)

    return topics


def extract_topics(curr_uttr):
    entities = get_entities(curr_uttr, only_named=True, with_labels=False)
    entities = [ent.lower() for ent in entities]
    entities = [ent for ent in entities
                if not (ent == "alexa" and curr_uttr["text"].lower()[:5] == "alexa") and "news" not in ent]
    if len(entities) == 0:
        for ent in get_entities(curr_uttr, only_named=False, with_labels=False):
            if ent in entities:
                pass
            else:
                entities.append(ent)
    entities = [ent for ent in entities if len(ent) > 0]
    return entities


def extract_topic_from_user_uttr(dialog):
    """
    Extract one of the considered topics out of `TOPIC_WORDS.keys()`.
    If none of them, return empty string.

    Args:
        dialog: dialog from agent

    Returns:
        string topic
    """
    entities = extract_topics(dialog["human_utterances"][-1])
    topics = []
    for entity in entities:
        topics += find_topics_in_substring(entity)
    if len(topics) > 0:
        logger.info(f"Extracted topic `{topics[-1]}` from user utterance.")
        return topics[-1]
    else:
        return ""


def pickup_topic_and_start_small_talk(dialog):
    """
    Pick up topic for small talk and return first response.

    Args:
        dialog: dialog from agent

    Returns:
        Tuple of (response, topic, confidence)
    """
    last_user_uttr = dialog["human_utterances"][-1]
    if len(dialog["bot_utterances"]) > 0:
        last_bot_uttr = dialog['bot_utterances'][-1]
    else:
        last_bot_uttr = {"text": "---", "annotations": {}}

    if if_choose_topic(last_user_uttr, last_bot_uttr) or if_switch_topic(last_user_uttr["text"].lower()):
        # user asks bot to chose topic: `pick up topic/what do you want to talk about/would you like to switch topic`
        # or bot asks user to chose topic and user says `nothing/anything/don't know`
        # if user asks to switch the topic
        topic = offer_topic(dialog)
        if topic in TOPIC_PATTERNS:
            if topic == "me":
                response = f"Let's talk about you. " + TOPIC_SCRIPTS.get(topic, [""])[0]
            elif topic == "you":
                response = f"Let's talk about me. " + TOPIC_SCRIPTS.get(topic, [""])[0]
            else:
                response = f"Let's talk about {topic}. " + TOPIC_SCRIPTS.get(topic, [""])[0]
            confidence = BOT_TOPIC_START_CONFIDENCE
        else:
            response = ""
            confidence = 0.
        logger.info(f"Bot initiates script on topic: `{topic}`.")
    elif if_chat_about_particular_topic(last_user_uttr, last_bot_uttr):
        # user said `let's talk about [topic]` or
        # bot said `what do you want to talk about/would you like to switch the topic`,
        #   and user answered [topic] (not something, nothing, i don't know - in this case,
        #   it will be gone through previous if)
        topic = extract_topic_from_user_uttr(dialog)
        if len(topic) > 0:
            response = TOPIC_SCRIPTS.get(topic, [""])[0]
            if topic in NOT_SCRIPTED_TOPICS:
                confidence = YES_CONTINUE_CONFIDENCE
            else:
                confidence = USER_TOPIC_START_CONFIDENCE
            logger.info(f"User initiates script on topic: `{topic}`.")
        else:
            response = ""
            confidence = 0.
            logger.info(f"Topic was not extracted.")
    else:
        topic = extract_topic_from_user_uttr(dialog)
        if len(topic) > 0:
            response = TOPIC_SCRIPTS.get(topic, [""])[0]
            confidence = FOUND_WORD_START_CONFIDENCE
            logger.info(f"Found word in user utterance on topic: `{topic}`.")
        else:
            topic = ""
            response = ""
            confidence = 0.

    return response, topic, confidence


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
