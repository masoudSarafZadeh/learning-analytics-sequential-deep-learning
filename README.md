# Learning Analytics & Sequential Deep Learning

This repository contains advanced sequential deep learning architectures designed to predict student performance and early dropout using the Open University Learning Analytics Dataset (OULAD). 

The project leverages both PyTorch and TensorFlow to process multi-modal sequential data (assessments, virtual learning environment (VLE) clickstreams, and demographics) through standard and novel neural network architectures.

## Repository Structure

* `data/`: Contains instructions for downloading the raw OULAD dataset.
* `src/`: Core source code and model definitions.
  * `preprocess.py`: Data cleaning, merging, and sequence generation.
  * `utils.py`: PyTorch Lightning DataModules and helper functions.
  * `lstm_pytorch.py`: Baseline sequential modeling using Long Short-Term Memory networks.
  * `TransformerEncoder_pytorch.py`: Causal Transformer architecture with custom Attention Pooling.
  * `composite_autoencoder_tensorflow.py`: Specialized semi-supervised bottleneck autoencoder.
  * `BERT_style.py`: (Roadmap) Bidirectional sequence representation.
  * `processed_data/`: Stores processed data (`df_all.csv`) and saved encoders (`label_encoder.joblib`).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/learning-analytics-sequential-deep-learning.git
   cd learning-analytics-sequential-deep-learning
   ```
2. **Install dependencies:**

```Bash
pip install -r requirements.txt
```
3. **Download Data:**
Follow the instructions in `data/README.md` to download and place the raw OULAD `.csv` files.

4. **Preprocess:**
Run the preprocessing script to generate the necessary artifacts.

```Bash
python src/preprocess.py
```
## Advanced Architectures (Methodology Notes)
This repository includes prototype code for methodologies currently under peer review for publication.

### Composite Autoencoder (`composite_autoencoder_tensorflow.py`)
This script demonstrates a **Semi-Supervised Composite Autoencoder** applied to the Assessment data stream. It forces a strict 2-neuron bottleneck to simultaneously:

1.  Reconstruct the input sequence (MSE Loss).

2.  Predict the final student outcome (Binary Crossentropy Loss).

**Note for Reproduction:** This repository provides the foundational implementation of our multi-stream composite autoencoder framework, demonstrated here on the Assessment data stream. The architecture is designed to be fully modular and symmetrically scales to companion domains (such as VLE clickstreams and Demographics). By compressing sequential features into a strict 2-dimensional bottleneck, the network isolates domain-specific latent representations. These dense, low-dimensional embeddings are structured to be concatenated for downstream fusion in a metamodel ensemble, maximizing predictive robustness while preventing overfitting.

### Bidirectional Transformer / BERT (`BERT_style.py`)
Currently marked for post-submission development. Future updates will introduce a Bidirectional Encoder Representations from Transformers (BERT) style architecture. This approach will utilize Masked Language Modeling (MLM) on student timelines, a dedicated `[CLS]` token for classification, and custom attention mechanisms designed specifically to maximize the interpretability of educational sequences.


## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
