import os
from flask import Flask, render_template, request
import numpy as np
import datasets
import transformers
import spacy
import re
import datetime


# Model path
models_path = './models/'
# Datasets path
datasets_path = './datasets/'
# Location of the record file
dataset_file_path = os.path.join(datasets_path, 'records.json')
# Number of example texts to load from surrey-nlp/PLOD-CW-25
n_example_texts = 20


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


# Model to host
model = None
tokenizer = None
trainer = None


def convert_ner_tags_to_int(dataset):
    dataset['ner_tag_ids'] = [tag_id_dict[tag] for tag in dataset['ner_tags']]
    return dataset


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


def diff_named_entity(tokens, true_tag_ids, pred_tag_ids):
    doc = spacy.blank('en')(' '.join(tokens))

    char_cnt = prev_char_cnt = 0
    prev_pred_tag_id = 0
    start_char_cnt = -1
    was_wrong = False
    ents = []
    for token, true_tag_id, pred_tag_id in zip(tokens, true_tag_ids, pred_tag_ids):
        token_len = len(token)
        if true_tag_id != pred_tag_id:
            # The model made the wrong prediction
            # Highlight everything wrong
            if was_wrong:
                # Highlight everything different
                if prev_pred_tag_id != pred_tag_id:
                    ents.append(doc.char_span(start_char_cnt, prev_char_cnt, '(' + tag_id_dict_reverse[prev_pred_tag_id] + ')'))
                    start_char_cnt = char_cnt
            else:
                if prev_pred_tag_id != 0:
                    ents.append(doc.char_span(start_char_cnt, prev_char_cnt, cls_id_dict_reverse[tag_cls_id_dict[prev_pred_tag_id]]))
                start_char_cnt = char_cnt
                was_wrong = True
        else:
            # The model made the correct prediction
            if was_wrong:
                ents.append(doc.char_span(start_char_cnt, prev_char_cnt, '(' + tag_id_dict_reverse[prev_pred_tag_id] + ')'))
                start_char_cnt = char_cnt
                was_wrong = False
            else:
                # Highlight by cls of predition
                if tag_cls_id_dict[pred_tag_id] != tag_cls_id_dict[prev_pred_tag_id]:
                    if prev_pred_tag_id != 0:
                        ents.append(doc.char_span(start_char_cnt, prev_char_cnt, cls_id_dict_reverse[tag_cls_id_dict[prev_pred_tag_id]]))
                    start_char_cnt = char_cnt
        prev_pred_tag_id = pred_tag_id
        prev_char_cnt = char_cnt + token_len
        char_cnt += token_len + 1

    if prev_pred_tag_id != 0:
        # Case where the sentence ends with non-O tag
        ents.append(doc.char_span(start_char_cnt, prev_char_cnt, cls_id_dict_reverse[tag_cls_id_dict[prev_pred_tag_id]]))
    elif was_wrong:
        ents.append(doc.char_span(start_char_cnt, prev_char_cnt, '(' + tag_id_dict_reverse[prev_pred_tag_id] + ')'))

    doc.ents = ents

    err_tag_colors = tag_colors | {'('+k+')':'red' for k in tag_colors.keys()} | {'(O)':'red'}
    return spacy.displacy.render(doc, style='ent', options={'colors': err_tag_colors})


# Pre-loaded examples
dataset_example = datasets.load_dataset('surrey-nlp/PLOD-CW-25', split=datasets.ReadInstruction('test', to=n_example_texts))
dataset_example = dataset_example.map(convert_ner_tags_to_int)
EXAMPLE_TEXTS = [{'id': idx+1, 'text': ' '.join(tokens)} for (idx, tokens) in enumerate(dataset_example['tokens'])]


def requestResults(usr_input):
    if type(usr_input) is int:
        # Use example texts
        text = EXAMPLE_TEXTS[usr_input]['text']
        tokens = dataset_example['tokens'][usr_input]
        true_tag_ids = dataset_example['ner_tag_ids'][usr_input]
    else:
        # Split the text into initial tokens
        text = usr_input
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        if len(tokens) == 0:
            # Empty text
            return {
                'text': text,
                'prediction': '',
            }
        true_tag_ids = [0 for token in tokens]

    # Create dataset object
    dataset_original = datasets.Dataset.from_dict({'tokens': [tokens], 'ner_tag_ids': [true_tag_ids]})
    dataset_tokenized = dataset_original.map(tokenize_and_align_tag_ids, batched=True)

    predictions, labels, _ = trainer.predict(dataset_tokenized)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_tag_ids = [
        [int(l) for (p, l) in zip(prediction, label) if l != ignore_id]
        for prediction, label in zip(predictions, labels)
    ][0]
    pred_tag_ids = [
        [int(p) for (p, l) in zip(prediction, label) if l != ignore_id]
        for prediction, label in zip(predictions, labels)
    ][0]

    # Add to record
    try:
        dataset_file = datasets.load_dataset("json", data_files=dataset_file_path, split='train')
    except FileNotFoundError:
        print(f'Dataset file "{dataset_file_path}" not found. Creating an empty dataset.')
        dataset_file = None

    dataset_new = datasets.Dataset.from_dict({'utc_time': [str(datetime.datetime.now(datetime.UTC))], 'text': [text], 'tokens': [tokens], 'pred_ner_tag_ids': [pred_tag_ids]})
    if dataset_file is None:
        dataset_file = dataset_new
    else:
        dataset_file = datasets.concatenate_datasets([dataset_file, dataset_new])
    dataset_file.to_json(dataset_file_path)

    # Return results
    if type(usr_input) is int:
        true_html = display_named_entity(tokens, true_tag_ids)
        pred_html = diff_named_entity(tokens, true_tag_ids, pred_tag_ids)
    else:
        true_html = None
        pred_html = display_named_entity(tokens, pred_tag_ids)
        text = ' '.join(tokens)
    return {
        'text': text,
        'ground_truth': true_html,
        'prediction': pred_html,
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
            usr_input = int(request.form['example_select']) - 1
        else:
            usr_input = request.form['search']

        results = requestResults(usr_input)
        return render_template('home.html', results=results, examples=EXAMPLE_TEXTS)


def main():
    # Prompt for model to host
    if not os.path.exists(models_path):
        print(f'Cant find any model to host. Please put the models into {models_path}. E.g.: {os.path.join(models_path, "bert-NER/*.*")}')
        return
    model_names = [item for item in sorted(os.listdir(models_path)) if os.path.isdir(os.path.join(models_path, item))]
    if len(model_names) == 0:
        print(f'Cant find any model to host. Please put the models into {models_path}. E.g.: {os.path.join(models_path, "bert-NER/*.*")}')
        return

    if len(model_names) == 1:
        model_path = os.path.join(models_path, model_names[0])
    else:
        while True:
            try:
                print('Please choose the model to host:')
                for idx, model_name in enumerate(model_names):
                    print(f'{idx+1}: {model_name}')
                choice = int(input('Choice: '))
                if choice <= 0 or choice > len(model_names):
                    raise ValueError()
                model_path = os.path.join(models_path, model_names[choice-1])
                break
            except ValueError:
                print('Invalid input. Please enter a valid integer.')
                print()

    global model, tokenizer, trainer
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    trainer = transformers.Trainer(model, data_collator=transformers.DataCollatorForTokenClassification(tokenizer))

    # Use debug=False to prevent prompting model choice twice
    app.run(debug=False)


if __name__ == '__main__':
    main()
