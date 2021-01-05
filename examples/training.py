import chaine
from chaine.data import Token
from flair.models import SequenceTagger
from flair.data import Sentence
import tqdm
import datasets

TAGGER = SequenceTagger.load("pos-multi-fast")
DATASET = datasets.load_dataset("germaner")


def preprocess(dataset):
    for tokens in tqdm.tqdm(dataset):
        sentence = Sentence(" ".join(tokens), use_tokenizer=False)
        TAGGER.predict(sentence)
        pos_tags = [token.get_tag("upos").value for token in sentence]
        features = [Token(i, text).features for i, text in enumerate(tokens)]
        for token, pos in zip(features, pos_tags):
            token["pos"] = pos
        yield features





if __name__ == "__main__":
    tokens = preprocess(DATASET["train"]["tokens"][:10])
    labels = DATASET["train"]["ner_tags"][:10]

    model = chaine.train(tokens, labels)
