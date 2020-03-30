import torch
import logging
from torch.utils.data import TensorDataset
from process.progressbar import ProgressBar


logger = logging.getLogger()
class processer():
    def __init__(self):
        pass
    def get_labels(self):
        return ['0','1']

    def read_txt(self,filename):
        with open(filename,'r') as rf:
            lines = rf.readlines()
        return lines

    def create_examples(self,data,type):
        examples = []
        for i,line in enumerate(data):
            guid = f'{i}-{line}'
            text_a = line.split('\t')[1]
            text_b = None
            label = line.split('\t')[3].replace('\n','') if type != 'test' else '0'
            example = InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label)
            examples.append(example)
        return examples

    def convert_examples_to_features(self,examples, tokenizer,
                                          max_length=512,
                                          label_list=None,
                                          output_mode=None,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True,
                                          split_num = 4):

        """
        Loads a data file into a list of ``InputFeatures``
        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            task: CLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d" % (ex_index))

                #***对长文本进行切分，将切分后的每一个子句作为一个单独完整的句子，计算feature***
                split_text_length = int(len(example.text_a) / split_num)
                split_features = []

                for i in range(split_num):
                    split_text = example.text_a[split_text_length * i:split_text_length * (i + 1)]

                    inputs = tokenizer.encode_plus(split_text,example.text_b,add_special_tokens=True,max_length=max_length)
                    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                    input_len = len(input_ids)
                    # Zero-pad up to the sequence length.
                    padding_length = max_length - len(input_ids)

                    if pad_on_left:
                        input_ids = ([pad_token] * padding_length) + input_ids
                        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                    else:
                        input_ids = input_ids + ([pad_token] * padding_length)
                        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
                    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                        max_length)
                    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                        max_length)
                    if output_mode == "classification":
                        label = label_map[example.label]
                    elif output_mode == "regression":
                        label = float(example.label)
                    else:
                        raise KeyError(output_mode)

                    if ex_index < 5:
                        logger.info("*** Example ***")
                        logger.info("guid: %s" % (example.guid))
                        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                        logger.info("label: %s (id = %s)" % (example.label, label))
                        logger.info("input length: %d" % (input_len))

                    split_features.append(InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label,
                              input_len=input_len))#split_features中包含的就是split_num个子句的InputFeatures对象

                features.append(split_features)

        return features

    def create_dataset(self,features):

        features_input_ids, features_attention_mask,features_token_type_ids,features_input_len,features_label= [],[],[],[],[]
        for split_features in features:
            split_features_input_ids, split_features_attention_mask,split_features_token_type_ids,split_features_input_len,split_features_label= [],[],[],[],[]
            
            split_features_input_ids.append(split_features_input_ids)
            split_features_attention_mask.append(split_features_attention_mask)
            split_features_token_type_ids.append(split_features_token_type_ids)
            split_features_input_len.append(split_features_input_len)
            split_features_label.append(split_features_label)

        features_input_ids.extend(split_features_input_ids)
        features_attention_mask.extend(split_features_attention_mask)
        features_token_type_ids.extend(split_features_token_type_ids)
        features_input_len.extend(split_features_input_len)
        features_attention_mask.extend(split_features_attention_mask)
    
    features_input_ids = torch.tensor(features_input_ids)
    features_attention_mask = torch.tensor(features_attention_mask)
    features_token_type_ids = torch.tensor(features_token_type_ids)
    features_input_len = torch.tensor(features_input_len)
    features_attention_mask = torch.tensor(features_attention_mask)
    
       
    print(all_input_ids.shape)
    print(all_attention_mask.shape)
    print(all_token_type_ids.shape)
    print(all_lens.shape)
    print(all_labels.shape)

    dataset = TensorDataset(features_input_ids, features_attention_mask, features_token_type_ids, features_input_len, features_attention_mask)
    return dataset

        #原版
        # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        # all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        # all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        # all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
        # return dataset





def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label,input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

