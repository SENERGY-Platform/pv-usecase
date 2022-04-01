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

__all__ = ("OperatorBase",)

from .logger import logger
import confluent_kafka
import mf_lib
import json
import typing


def log_kafka_sub_action(action: str, partitions: typing.List):
    for partition in partitions:
        logger.info(
            f"subscription event: action={action} topic={partition.topic} partition={partition.partition} offset={partition.offset}"
        )


def on_assign(_, p):
    log_kafka_sub_action("assign", p)


def on_revoke(_, p):
    log_kafka_sub_action("revoke", p)


def on_lost(_, p):
    log_kafka_sub_action("lost", p)


class OperatorBase:
    def __init__(self, kafka_consumer: confluent_kafka.Consumer, kafka_producer: confluent_kafka.Producer, filter_handler: mf_lib.FilterHandler, poll_timeout: float = 1.0):
        self.__kafka_consumer = kafka_consumer
        self.__kafka_producer = kafka_producer
        self.__filter_handler = filter_handler
        self.__poll_timeout = poll_timeout
        self.__stop = False
        self.__stopped = False

    def __run(self):
        sources = self.__filter_handler.get_sources()
        if sources:
            self.__kafka_consumer.subscribe(
                sources,
                on_assign=on_assign,
                on_revoke=on_revoke,
                on_lost=on_lost
            )
        else:
            raise RuntimeError("no sources")
        while not self.__stop:
            try:
                msg_obj = self.__kafka_consumer.poll(timeout=self.__poll_timeout)
                if msg_obj:
                    if not msg_obj.error():
                        try:
                            for result in self.__filter_handler.get_results(message=json.loads(msg_obj.value())):
                                if not result.ex:
                                    for f_id in result.filter_ids:
                                        self.run(
                                            selector=self.__filter_handler.get_filter_args(id=f_id)["selector"],
                                            data=result.data
                                        )
                                else:
                                    logger.error(result.ex)
                        except mf_lib.exceptions.NoFilterError:
                            pass
                        except mf_lib.exceptions.MessageIdentificationError as ex:
                            logger.error(ex)
                    else:
                        raise confluent_kafka.KafkaException(msg_obj.error())
            except Exception as ex:
                logger.exception(ex)
                self.__stop = True
        self.__stopped = True

    def start(self):
        self.__run()

    def stop(self):
        self.__stop = True

    def is_alive(self):
        return self.__stopped

    def run(self, selector, data):
        pass
