# Fine Tuning Examples (easy peasy) 👷‍♂️👷‍♀️

| Dataset                                          | Fine Tuning Example                                          |
| ------------------------------------------------ | ------------------------------------------------------------ |
| Fine Tune on Mozilla Turkish Dataset             | <a href="https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Fine-tuning Example&logo=Google%20Colab&color=f9ab00"></a> |
| Sample Code for Other Dataset And other Language | [github_link](https://github.com/m3hrdadfi/notebooks/)       |





## How to use yourself❓

if you check ***`Fine Tune on Mozilla Turkish Dataset`*** link above you can use this model **but be careful change the model and processor name to my model** 

EXAMPLE : 

Change this 

```python
model = Wav2Vec2ForCTC.from_pretrained(
"facebook/wav2vec2-large-xlsr-53",
attention_dropout=0.1,
hidden_dropout=0.1,
feat_proj_dropout=0.0,
mask_time_prob=0.05,
layerdrop=0.1,
ctc_loss_reduction="mean",
pad_token_id=processor.tokenizer.pad_token_id,
vocab_size=len(processor.tokenizer)
)

```

to  this : 

```python
model = Wav2Vec2ForCTC.from_pretrained(
"masoudmzb/wav2vec2-xlsr-multilingual-53-fa",
attention_dropout=0.1,
hidden_dropout=0.1,
feat_proj_dropout=0.0,
mask_time_prob=0.05,
layerdrop=0.1,
ctc_loss_reduction="mean",
pad_token_id=processor.tokenizer.pad_token_id,
vocab_size=len(processor.tokenizer)
)

```

and same for processor .change this :

```bash
from transformers import Wav2Vec2Processor
model_name_or_path = "facebook/wav2vec2-large-xlsr-53"
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)

```

to this : 

```bash
from transformers import Wav2Vec2Processor
model_name_or_path = "masoudmzb/wav2vec2-xlsr-multilingual-53-fa"
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
```

Now you have all you need.



### Use FineTuned Model

This model is finetuned on [m3hrdadfi/wav2vec2-large-xlsr-persian-v3](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3) , so the process for train or evaluation is same

> ```bash
> # requirement packages
> !pip install git+https://github.com/huggingface/datasets.git
> !pip install git+https://github.com/huggingface/transformers.git
> !pip install torchaudio
> !pip install librosa
> !pip install jiwer
> !pip install parsivar
> !pip install num2fawords
> ```



**Normalizer**

```bash
# Normalizer
!wget -O normalizer.py https://huggingface.co/m3hrdadfi/"wav2vec2-large-xlsr-persian-v3/raw/main/dictionary.py
!wget -O normalizer.py https://huggingface.co/m3hrdadfi/"wav2vec2-large-xlsr-persian-v3/raw/main/normalizer.py

```



If you are not sure your transcriptions are clean or not (having weird characters or any other alphabete chars ) use this code provided by  [m3hrdadfi/wav2vec2-large-xlsr-persian-v3](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3)  



**Cleaning** (Fill the data part with your own data dir)

```python

from normalizer import normalizer

def cleaning(text):
    if not isinstance(text, str):
        return None

    return normalizer({"sentence": text}, return_dict=False)

# edit these parts with your own data directory

data_dir = "data"


test = pd.read_csv(f"{data_dir}/yourtest.tsv", sep="	")
test["path"] = data_dir + "/clips/" + test["path"]
print(f"Step 0: {len(test)}")

test["status"] = test["path"].apply(lambda path: True if os.path.exists(path) else None)
test = test.dropna(subset=["path"])
test = test.drop("status", 1)
print(f"Step 1: {len(test)}")

test["sentence"] = test["sentence"].apply(lambda t: cleaning(t))
test = test.dropna(subset=["sentence"])
print(f"Step 2: {len(test)}")

test = test.reset_index(drop=True)
print(test.head())

test = test[["path", "sentence"]]
test.to_csv("/content/test.csv", sep="	", encoding="utf-8", index=False)
```



**Prediction**

```python
import numpy as np
import pandas as pd

import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, load_metric

import IPython.display as ipd

model_name_or_path = "masoudmzb/wav2vec2-xlsr-multilingual-53-fa"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model_name_or_path, device)

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = processor(
        batch["speech"], 
        sampling_rate=processor.feature_extractor.sampling_rate, 
        return_tensors="pt", 
        padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits 

    pred_ids = torch.argmax(logits, dim=-1)

    batch["predicted"] = processor.batch_decode(pred_ids)
    return batch

# edit these parts with your own data directory
dataset = load_dataset("csv", data_files={"test": "/path_to/your_test.csv"}, delimiter="	")["test"]
dataset = dataset.map(speech_file_to_array_fn)
result = dataset.map(predict, batched=True, batch_size=4)
```



**WER Score**

```python

wer = load_metric("wer")
print("WER: {:.2f}".format(100 * wer.compute(predictions=result["predicted"], references=result["sentence"])))
```



**Output**

```python

max_items = np.random.randint(0, len(result), 20).tolist()
for i in max_items:
    reference, predicted =  result["sentence"][i], result["predicted"][i]
    print("reference:", reference)
    print("predicted:", predicted)
    print('---')
```





## training details: 🔭

One model was trained on Persian Mozilla dataset before So we Decided to continue from that one. Model is warm started from `mehrdadfa`’s [checkpoint](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3)  
- For more details, you can take a look at config.json at the model card in 🤗 Model Hub
- The model trained 84000 steps, equal to  12.42 Epochs.
- The base model to finetune was https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3/tree/main

## Fine Tuning Recommendations: 🐤
For fine tuning you can check the link below. but be aware some Tips. you may need gradient_accumulation because you need more batch size.  the are many hyperparameters make sure you set them properly : 

- learning_rate
- attention_dropout
-  hidden_dropout
-  feat_proj_dropout
-  mask_time_prob
-  layer_drop



