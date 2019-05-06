from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from pretrained_bert import *

class BertQAPredictorTest(AllenNlpTestCase):
    def test_predict(self):
        model_url = "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/bert-base/squad1.1/model.tar.gz"
        archive = load_archive(model_url)
        predictor = Predictor.from_archive(archive, 'bert-for-qa')
        passage = """Robots can be used in any situation and for any purpose, but today many are used in dangerous
                     environments (including bomb detection and de-activation), manufacturing processes, or where
                     humans cannot survive."""
        question = "What can be used in dangerous environments?"
        prediction = predictor.predict(question, passage)
        assert prediction["predictions"] == "Robots"
