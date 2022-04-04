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

from ._util import *
import unittest


class TestOperator(unittest.TestCase):
    def test_call_run(self):
        with open("tests/resources/mock_opr_config.json") as file:
            mock_opr_config = json.load(file)
        filter_handler = init_filter_handler(mock_opr_config)
        mock_operator = MockOperator()
        mock_operator.init(
            kafka_consumer=None,
            kafka_producer=None,
            filter_handler=filter_handler,
            output_topic=None,
            pipeline_id=None,
            operator_id=None
        )
        for message in mock_messages:
            mock_operator._OperatorBase__call_run(message)


if __name__ == '__main__':
    unittest.main()
