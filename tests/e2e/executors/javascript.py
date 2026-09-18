import json

from m2cgen import export_to_javascript

from tests import utils
from tests.e2e.executors.base import BaseExecutor

# Reads the sample values from the command line arguments and prints the
# score in JSON format (a number, or an array for multi-output models).
RUNNER = """
const args = process.argv.slice(2).map(Number);
const res = score(args);
console.log(JSON.stringify(res));
"""


class JavascriptExecutor(BaseExecutor):

    def __init__(self, model):
        self.model = model

        self.script_path = None

    def predict(self, X):
        args = map(utils.format_arg, X)

        result = utils.execute_command(["node", str(self.script_path), *args])

        result = json.loads(result)
        if isinstance(result, list):
            return result
        return [result]

    def prepare(self):
        code = export_to_javascript(self.model) + RUNNER

        self.script_path = self._resource_tmp_dir / "model.js"
        utils.write_content_to_file(code, self.script_path)
