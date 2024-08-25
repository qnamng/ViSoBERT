# <a name="introduction"></a> ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing (EMNLP 2023 - Main)
**Disclaimer**: The paper contains actual comments on social networks that might be construed as abusive, offensive, or obscene.

ViSoBERT is the state-of-the-art language model for Vietnamese social media tasks:

 - ViSoBERT is the first monolingual MLM ([XLM-R](https://github.com/facebookresearch/XLM#xlm-r-new-model) architecture) built specifically for Vietnamese social media texts.
 - ViSoBERT outperforms previous monolingual, multilingual, and multilingual social media approaches, obtaining new state-of-the-art performances on four downstream Vietnamese social media tasks.

The general architecture and experimental results of ViSoBERT can be found in our [paper](https://aclanthology.org/2023.emnlp-main.315/):

    @inproceedings{nguyen-etal-2023-visobert,
        title = "{V}i{S}o{BERT}: A Pre-Trained Language Model for {V}ietnamese Social Media Text Processing",
        author = "Nguyen, Nam  and
          Phan, Thang  and
          Nguyen, Duc-Vu  and
          Nguyen, Kiet",
        editor = "Bouamor, Houda  and
          Pino, Juan  and
          Bali, Kalika",
        booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
        month = dec,
        year = "2023",
        address = "Singapore",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.emnlp-main.315",
        pages = "5191--5207",
        abstract = "English and Chinese, known as resource-rich languages, have witnessed the strong development of transformer-based language models for natural language processing tasks. Although Vietnam has approximately 100M people speaking Vietnamese, several pre-trained models, e.g., PhoBERT, ViBERT, and vELECTRA, performed well on general Vietnamese NLP tasks, including POS tagging and named entity recognition. These pre-trained language models are still limited to Vietnamese social media tasks. In this paper, we present the first monolingual pre-trained language model for Vietnamese social media texts, ViSoBERT, which is pre-trained on a large-scale corpus of high-quality and diverse Vietnamese social media texts using XLM-R architecture. Moreover, we explored our pre-trained model on five important natural language downstream tasks on Vietnamese social media texts: emotion recognition, hate speech detection, sentiment analysis, spam reviews detection, and hate speech spans detection. Our experiments demonstrate that ViSoBERT, with far fewer parameters, surpasses the previous state-of-the-art models on multiple Vietnamese social media tasks. Our ViSoBERT model is available only for research purposes. Disclaimer: This paper contains actual comments on social networks that might be construed as abusive, offensive, or obscene.",
    }
    
The pretraining dataset of our paper is available at: [Pretraining dataset](https://drive.google.com/drive/folders/1C144LOlkbH78m0-JoMckpRXubV7XT7Kb)

**Please CITE** our paper when ViSoBERT is used to help produce published results or is incorporated into other software.

**Installation** 

Install `transformers` and `SentencePiece` packages:
    
    pip install transformers
    pip install SentencePiece

**Example usage**
```python
from transformers import AutoModel, AutoTokenizer
import torch

model= AutoModel.from_pretrained('uitnlp/visobert')
tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')

encoding = tokenizer('hào quang rực rỡ', return_tensors='pt')

with torch.no_grad():
  output = model(**encoding)
```
