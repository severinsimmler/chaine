import re


class _Message:
    def __init__(self):
        self.iteration = None
        self.loss = None

    def __str__(self) -> str:
        return f"Iteration: {self.iteration}\tLoss: {self.loss}"


class _LogParser:
    def __init__(self):
        self.message = _Message()

    def parse(self, line: str) -> str:
        if (m := re.match(r"\*{5} (?:Iteration|Epoch) #(\d+) \*{5}\n", line)) :
            self.message.iteration = m.group(1)
        elif (m := re.match(r"Loss: (\d+\.\d+)", line)) :
            self.message.loss = m.group(1)
            text = str(self.message)
            self.message = _Message()
            return text
