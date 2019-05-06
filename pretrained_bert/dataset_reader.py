import json
import logging
import collections
from typing import List

import torch
from overrides import overrides
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.tokenization import whitespace_tokenize

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("squad_for_pretrained_bert")
class BertQADatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 pretrained_bert_model_file: str = 'bert-base-uncased',
                 max_query_length: int = 64,
                 max_sequence_length: int = 384,
                 document_stride: int = 128) -> None:
        super().__init__(lazy)
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model_file)
        self._max_query_length = max_query_length
        self._max_sequence_length = max_sequence_length
        self._document_stride = document_stride

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
            for entry in dataset:
                for paragraph in entry["paragraphs"]:
                    paragraph_text = paragraph["context"]
                    for question_answer_pair in paragraph["qas"]:
                        question_text = question_answer_pair["question"]
                        answer = None
                        answer_start_index = None
                        is_impossible = False
                        if "is_impossible" in question_answer_pair:
                            is_impossible = question_answer_pair["is_impossible"]
                        if not is_impossible and "answers" in question_answer_pair:
                            answer_info = question_answer_pair["answers"][0]
                            answer = answer_info["text"]
                            answer_start_index = answer_info["answer_start"]
                        instance = self.text_to_instance(question_text=question_text,
                                                         paragraph_text=paragraph_text,
                                                         answer=answer,
                                                         answer_start_index=answer_start_index,
                                                         is_impossible=is_impossible)
                        if instance is not None:
                            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         paragraph_text: str,
                         answer: str = None,
                         answer_start_index: int = None,
                         is_impossible: bool = False) -> List[Instance]:
        # pylint: disable=arguments-differ
        # TODO (pradeep): We're ignoring the instances where the paragraph text is longer than the limit set in the
        # constructor. HuggingFace's code makes multiple instances in that case. Should think of a better way to
        # handle long paragraphs here.
        def is_whitespace(char):
            if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
                return True
            return False

        doc_tokens: List[str] = []
        character_to_word_offset: List[int] = []
        prev_is_whitespace = True
        for char in paragraph_text:
            if is_whitespace(char):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(char)
                else:
                    doc_tokens[-1] += char
                prev_is_whitespace = False
            character_to_word_offset.append(len(doc_tokens) - 1)

        if answer is None:
            answer_start_position = None
            answer_end_position = None
        elif not is_impossible:
            answer_start_position = character_to_word_offset[answer_start_index]
            answer_end_position = character_to_word_offset[answer_start_index + len(answer) - 1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[answer_start_position : (answer_end_position + 1)])
            cleaned_answer_text = " ".join(
                whitespace_tokenize(answer))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Skipping instance."
                               f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
                return None
        else:
            answer_start_position = -1
            answer_end_position = -1 

        query_tokens = self._tokenizer.tokenize(question_text)

        if len(query_tokens) > self._max_query_length:
            query_tokens = query_tokens[0:self._max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self._max_sequence_length - len(query_tokens) - 3

        if len(all_doc_tokens) > max_tokens_for_doc:
            logger.warn("Ignoring instance longer than maximum allowed length."
                        f"Paragraph had {len(all_doc_tokens)} and maximum allowed length is {max_tokens_for_doc}")
            return None


        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(len(all_doc_tokens)):
            split_token_index = i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self._max_sequence_length
        assert len(input_mask) == self._max_sequence_length
        assert len(segment_ids) == self._max_sequence_length
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
        segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)

        fields = {"input_ids": MetadataField(input_ids_tensor),
                  "token_type_ids": MetadataField(segment_ids_tensor),
                  "attention_mask": MetadataField(input_mask_tensor),
                  "tokens": MetadataField(tokens),
                  "document_tokens": MetadataField(doc_tokens),
                  "token_to_original_map": MetadataField(token_to_orig_map)}

        if answer_start_position is not None and answer_end_position is not None:
            answer_start_position_tensor = torch.tensor(answer_start_position, dtype=torch.long)
            answer_end_position_tensor = torch.tensor(answer_end_position, dtype=torch.long)
            fields["answer_start_position"] = MetadataField(answer_start_position_tensor)
            fields["answer_end_position"] = MetadataField(answer_end_position_tensor)

        instance = Instance(fields)
        return instance
