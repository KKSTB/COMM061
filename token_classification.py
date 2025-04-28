from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import datasets
import transformers
import spacy
import re


# Define named entities
ignore_id = -100

tag_list = ['O', 'B-AC', 'B-LF', 'I-LF']
tag_id_dict = {k:v for v,k in enumerate(tag_list)}
tag_id_dict_reverse = {v:k for k,v in tag_id_dict.items()}

tag_cls_dict = {'O': 'O', 'B-AC': 'AC', 'B-LF': 'LF', 'I-LF': 'LF'}
cls_list = dict.fromkeys(tag_cls_dict.values()).keys()
cls_id_dict = {k:v for v,k in enumerate(cls_list)}
cls_id_dict_reverse = {v:k for k,v in cls_id_dict.items()}

tag_cls_id_dict = {tag_id_dict[k]:cls_id_dict[tag_cls_dict[k]] for k in tag_list}

n_tags = len(tag_list)
n_cls = len(cls_list)

tag_colors = {
    'B-AC': 'yellow',
    'B-LF': 'green',
    'I-LF': 'green',
    'AC': 'yellow',
    'LF': 'green',
}


model = transformers.AutoModelForTokenClassification.from_pretrained('./bert_8000/')
tokenizer = transformers.AutoTokenizer.from_pretrained('./bert_8000/')
trainer = transformers.Trainer(model, data_collator=transformers.DataCollatorForTokenClassification(tokenizer))


def tokenize_and_align_tag_ids(dataset):
    tokenized_inputs = tokenizer(dataset['tokens'], truncation=True, is_split_into_words=True)

    all_tokenized_tag_ids = []
    for i, word_tag_ids in enumerate(dataset['ner_tag_ids']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        tokenized_tag_ids = []
        for word_idx in word_ids:
            # Special tokens(i.e.[CLS],[SEP]) have a word id that is None. We set the label to ignore_id so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                tokenized_tag_ids.append(ignore_id)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                tokenized_tag_ids.append(word_tag_ids[word_idx])
            # For the other tokens in a word, we set the label to either the current label or ignore_id, depending on
            # the label_all_tokens flag.
            else:
                tokenized_tag_ids.append(ignore_id)
            previous_word_idx = word_idx

        all_tokenized_tag_ids.append(tokenized_tag_ids)

    tokenized_inputs['labels'] = all_tokenized_tag_ids
    return tokenized_inputs


def display_named_entity(tokens, tag_ids):
    doc = spacy.blank('en')(' '.join(tokens))

    char_cnt = 0
    prev_char_cnt = 0
    prev_tag_id = 0
    start_char_cnt = -1
    was_wrong = False
    ents = []
    for token, tag_id in zip(tokens, tag_ids):
        token_len = len(token)
        tag = tag_id_dict_reverse[tag_id]
        if tag.startswith('I'):
            # Check whether it is continued from the same class as the previous
            if tag_cls_id_dict[tag_id] != tag_cls_id_dict[prev_tag_id]:
                # Not the same class (the model made the wrong prediction)
                if prev_tag_id != 0:
                    if was_wrong:
                        ents.append(doc.char_span(start_char_cnt, prev_char_cnt, tag_id_dict_reverse[prev_tag_id]))
                    else:
                        ents.append(doc.char_span(start_char_cnt, prev_char_cnt, cls_id_dict_reverse[tag_cls_id_dict[prev_tag_id]]))
                start_char_cnt = char_cnt
                was_wrong = True
        elif tag.startswith('B'):
            # B tag
            if prev_tag_id == 0:
                # Begins from O tag
                pass
            else:
                # Begins from a different B tag
                if was_wrong:
                    ents.append(doc.char_span(start_char_cnt, prev_char_cnt, tag_id_dict_reverse[prev_tag_id]))
                else:
                    ents.append(doc.char_span(start_char_cnt, prev_char_cnt, cls_id_dict_reverse[tag_cls_id_dict[prev_tag_id]]))
            start_char_cnt = char_cnt
            was_wrong = False
        else:
            # O tag
            if prev_tag_id != 0:
                if was_wrong:
                    ents.append(doc.char_span(start_char_cnt, prev_char_cnt, tag_id_dict_reverse[prev_tag_id]))
                else:
                    ents.append(doc.char_span(start_char_cnt, prev_char_cnt, cls_id_dict_reverse[tag_cls_id_dict[prev_tag_id]]))
        prev_tag_id = tag_id
        prev_char_cnt = char_cnt + token_len
        char_cnt += token_len + 1

    if prev_tag_id != 0:
        # Case where the sentence ends with non-O tag
        if was_wrong:
            ents.append(doc.char_span(start_char_cnt, prev_char_cnt, tag_id_dict_reverse[prev_tag_id]))
        else:
            ents.append(doc.char_span(start_char_cnt, prev_char_cnt, cls_id_dict_reverse[tag_cls_id_dict[prev_tag_id]]))

    doc.ents = ents

    return spacy.displacy.render(doc, style='ent', options={'colors': tag_colors})


# Pre-loaded examples
EXAMPLE_TEXTS = [
    {
        'id': 1,
        'text': 'something called hydrogen deuterium exchange mass spectrometry (HDX-MS)'
    },
]


def requestResults(text):
    # Split the text into initial tokens
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    tag_ids = [0 for token in tokens]

    # Create dataset object
    dataset_original = pd.DataFrame({'tokens': [tokens], 'ner_tag_ids': [tag_ids]})
    dataset_original = datasets.Dataset.from_pandas(dataset_original)
    dataset_tokenized = dataset_original.map(tokenize_and_align_tag_ids, batched=True)

    predictions, labels, _ = trainer.predict(dataset_tokenized)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    pred_tag_ids = [
        [int(p) for (p, l) in zip(prediction, label) if l != ignore_id]
        for prediction, label in zip(predictions, labels)
    ]

    html = display_named_entity(tokens, pred_tag_ids[0])
    text = ' '.join(tokens)
    return {
        'text': text,
        'prediction': html,
    }


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html', examples=EXAMPLE_TEXTS)


@app.route('/', methods=['POST'])
def get_data():
    if request.method == 'POST':
        # Check if using example or manual input
        if 'example_select' in request.form and request.form['example_select'] != 'manual':
            text = request.form['example_select']
        else:
            text = request.form['search']
        
        results = requestResults(text)
        return render_template('home.html', results=results, examples=EXAMPLE_TEXTS)


if __name__ == '__main__':
    app.run(debug=True)
