# Latent-semantic-indexing
A small project analysing different embedding methodologies to perform latent semantic indexing in a information retrieval system. 

## Installation

Install the package

```bash
pip install git+https://github.com/dario-coscia/latent-semantic-indexing.git
```

Install the required Python libraries

```bash
pip3 install -qr requirements.txt
```

Download NLTK corpora (the ```reuters``` corpus is needed only to run the demo)

```python
import nltk
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')
nltk.download('reuters')
```

## Usage

After installing the required libraries, you can run the demo:

```python
python demo.py
```

## Author

- [Dario Coscia](https://github.com/dario-coscia)

## License

This project is licensed under the [MIT license](LICENSE).
