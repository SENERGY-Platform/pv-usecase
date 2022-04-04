"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import util
import mf_lib
import logging
import json

logger = logging.getLogger("operator")
logger.disabled = True

with open("tests/resources/mock_messages.json") as file:
    mock_messages = json.load(file)

with open("tests/resources/mock_opr_config.json") as file:
    mock_opr_config = json.load(file)

with open("tests/resources/mock_result.json") as file:
    mock_result = json.load(file)


def init_filter_handler(opr_config):
    if not isinstance(opr_config, util.OperatorConfig):
        opr_config = util.OperatorConfig(opr_config)
    filter_handler = mf_lib.FilterHandler()
    for it in opr_config.inputTopics:
        filter_handler.add_filter(util.gen_filter(input_topic=it, selectors=opr_config.config.selectors))
    return filter_handler


class MockOperator(util.OperatorBase):
    def func_1(self, a, timestamp):
        assert a == mock_messages[0]["data"]["val_a"]
        assert timestamp == mock_messages[0]["data"]["time"]
        return {"result": 1}

    def func_2(self, a, b, timestamp):
        assert a == mock_messages[1]["data"]["val_a"]
        assert b == mock_messages[1]["data"]["val_b"]
        assert timestamp == mock_messages[1]["data"]["time"]

    def run(self, data, selector):
        return getattr(self, selector)(**data)


class MockKafkaProducer:
    def __init__(self, result):
        self.__result = result
        self.__count = 0

    def produce(self, topic, value, key):
        assert self.__count < 1
        assert topic == self.__result["topic"]
        assert key == self.__result["key"]
        assert isinstance(value, str)
        value = json.loads(value)
        assert set(value) == set(self.__result["value"])
        assert value["pipeline_id"] == self.__result["value"]["pipeline_id"]
        assert value["operator_id"] == self.__result["value"]["operator_id"]
        assert isinstance(value["analytics"], dict)
        assert isinstance(value["time"], str)
        self.__count += 1
