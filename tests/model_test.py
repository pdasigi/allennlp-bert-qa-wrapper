from flaky import flaky

from allennlp.common.testing import ModelTestCase

from pretrained_bert import *


class BertQAModelTest(ModelTestCase):
    def setUp(self):
        super(BertQAModelTest, self).setUp()
        self.set_up_model("fixtures/model/experiment.json",
                          "fixtures/data/sample_data.json")

    @flaky
    def test_model_can_train_save_and_load(self):
        # Model does not use pooled outputs. We can ignore their gradients.
        gradients_to_ignore = {'bert_qa_model.bert.pooler.dense.weight',
                               'bert_qa_model.bert.pooler.dense.bias'}
        self.ensure_model_can_train_save_and_load(self.param_file,
                                                  gradients_to_ignore=gradients_to_ignore)
