from async_processor import (
    InputMessage,
    Processor
)


class MultiplicationProcessor(Processor):
    def process(self, input_message: InputMessage) -> int:
        body = input_message.body
        return body["x"] * body["y"]


app = MultiplicationProcessor().build_app()