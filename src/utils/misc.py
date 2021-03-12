# Source: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
import sys
from io import StringIO

class capture(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout