Smart MCQ Solver — DLGenAI

A deep learning and retrieval-based approach to solving multiple-choice questions by ranking the answer options.

This project was developed for the Smart MCQ Solver Challenge as part of the DLGenAI project. The goal was to predict the correct answer for each multiple-choice question and return the top 3 ranked options, which are evaluated using MAP@3.

The project explores three different approaches:

1. TF-IDF + PyTorch MLP
2. FAISS Retrieval + Fine-tuned DeBERTa
3. FAISS Retrieval + Cross-Encoder Reranking

The experiments compare a lightweight model built from scratch against pretrained transformer and retrieval-based approaches.

---

Problem

Each question contains:

- A question/prompt
- Five possible answers: "A", "B", "C", "D", "E"
- The correct answer in the training data

Example structure:

id | prompt | A | B | C | D | E | answer

The model is not simply required to predict one class. Instead, it must rank the five options and return the top three.

For example:

A E D

means:

1. A is ranked first
2. E is ranked second
3. D is ranked third

This makes MAP@3 the main evaluation metric.

---

Dataset

The training dataset contains 2,000 questions and 8 columns.

The columns are:

Column| Description
"id"| Question identifier
"prompt"| Question text
"A"| Option A
"B"| Option B
"C"| Option C
"D"| Option D
"E"| Option E
"answer"| Correct option

The notebook's initial analysis found:

- 2,000 training examples
- No missing values
- Five possible answer labels: "A", "B", "C", "D", "E"

The answer distribution was:

Answer| Count
A| 369
B| 490
C| 459
D| 358
E| 324

The data was split using an 80/20 stratified train-validation split with "random_state=42".

This resulted in:

- 1,600 training questions
- 400 validation questions

Because every question produces five question-option pairs, the first model used:

- 8,000 training pairs
- 2,000 validation pairs

---

Model 1 — TF-IDF + PyTorch MLP

The first approach was built from scratch rather than using a pretrained language model.

Feature construction

Each question was paired independently with each of its five options:

question + option

For every question, the correct option receives label "1" and the remaining four options receive label "0".

For example:

Question + A → 0
Question + B → 1
Question + C → 0
Question + D → 0
Question + E → 0

TF-IDF

The combined text was converted into TF-IDF features using:

- Maximum features: "10,000"
- English stop-word removal
- Unigrams + bigrams

TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

The resulting vectors were converted into PyTorch tensors.

PyTorch architecture

The neural network is a three-layer MLP:

Input: 10,000
       ↓
Linear(10000 → 512)
       ↓
ReLU
       ↓
Dropout(0.3)
       ↓
Linear(512 → 128)
       ↓
ReLU
       ↓
Dropout(0.3)
       ↓
Linear(128 → 1)

The model uses:

BCEWithLogitsLoss()

and the Adam optimizer with a learning rate of "0.001".

Training was performed for 20 epochs, while saving the model whenever validation loss improved.

Prediction

The model produces a score for each question-option pair.

The five options are then ranked by their scores, and the top three are returned.

---

Model 1 Results

Validation

The notebook reports:

Validation MAP@3: 0.9671

Top-1 accuracy on the validation set was:

94.75%

Kaggle

Public leaderboard score:

MAP@3: 0.74854

The large gap between local validation performance and the public leaderboard score is an important observation from the experiment.

The model therefore performed extremely well on the particular validation split, but the leaderboard result was substantially lower.

---

Model 2 — FAISS + DeBERTa

The second approach moved from a traditional TF-IDF representation to a retrieval-augmented transformer architecture.

Architecture

Question
   ↓
Dense Retrieval
   ↓
FAISS
   ↓
Top similar training examples
   ↓
Context
   ↓
DeBERTa-v3-base
   ↓
Score each answer option
   ↓
Rank options
   ↓
Top 3

The retrieval stage uses a dense embedding representation and FAISS to find semantically similar training examples.

The retrieved examples are then used as additional context for the question.

Retriever

The notebook experiments with Hugging Face embeddings and FAISS.

The intended retrieval setup uses dense semantic representations and cosine similarity.

The retrieved documents contain information in the form of:

Question: ...
Answer: ...

The retrieval stage was configured to retrieve the most similar examples before scoring the answer choices.

DeBERTa

The transformer used was:

microsoft/deberta-v3-base

The model was configured as a binary relevance scorer:

AutoModelForSequenceClassification(
    ...,
    num_labels=1
)

Each question-option pair is treated as a binary classification problem:

1 → correct option
0 → incorrect option

The model was trained using:

BCEWithLogitsLoss

with AdamW.

The training experiments used a learning rate of:

1e-5

and three epochs.

Inference

For each question:

1. Retrieve similar examples.
2. Concatenate the retrieved examples into context.
3. Combine the question and context with each of the five options.
4. Score all five options using DeBERTa.
5. Rank the options.
6. Return the top three.

Model 2 Result

The notebook reports a Kaggle public MAP@3 score of:

0.74438

---

Model 3 — FAISS + Cross-Encoder

The third approach explored a pretrained cross-encoder as a reranking model.

Architecture

Question
   ↓
FAISS Dense Retrieval
   ↓
Top similar examples
   ↓
Retrieved context
   ↓
Cross-Encoder
   ↓
Score question-option pairs
   ↓
Rank
   ↓
Top 3

The cross-encoder used was:

cross-encoder/ms-marco-MiniLM-L-6-v2

This model is designed for query-document relevance scoring.

Instead of independently embedding the question and option, the cross-encoder jointly processes the pair and produces a relevance score.

Retrieval

The same general FAISS-based retrieval idea from Model 2 was used.

The pipeline retrieved the top relevant examples and then used the cross-encoder to score candidate answers.

Fine-tuning

The notebook contains experiments for fine-tuning the cross-encoder, but the final approach was not fine-tuned because of GPU/resource constraints.

The notebook explicitly notes:

«"cant finetune due to GPU constraint cant risk my final submission"»

Model 3 Result

The notebook reports:

Kaggle MAP@3: 0.53325

---

Model Comparison

The experiments produced the following results:

Model| Architecture| Approach| Validation MAP@3| Kaggle MAP@3
Model 1| TF-IDF + MLP| From scratch| ~0.967| 0.74854
Model 2| FAISS + DeBERTa| RAG + Transformer| ~0.74| 0.74438
Model 3| FAISS + Cross-Encoder| RAG + Reranking| ~0.70| 0.53325

The leaderboard results show that the lightweight TF-IDF + MLP approach slightly outperformed the DeBERTa-based approach in this competition setup, while the cross-encoder experiment performed considerably worse.

---

Experiment Tracking

The project uses Weights & Biases (W&B) for experiment tracking.

The notebook logs information such as:

- Model configuration
- Training loss
- Validation metrics
- MAP@3
- Kaggle public leaderboard scores

This makes it possible to compare different experiments rather than relying only on the final submission.

---

Evaluation Metric

The main competition metric is Mean Average Precision at 3 (MAP@3).

For an individual question:

- Correct answer ranked 1st → score = "1"
- Correct answer ranked 2nd → score = "1/2"
- Correct answer ranked 3rd → score = "1/3"
- Correct answer outside the top 3 → score = "0"

The notebook implements this manually using "apk()" and "mapk()".

Conceptually:

Correct option at #1 → 1.000
Correct option at #2 → 0.500
Correct option at #3 → 0.333
Not in top 3        → 0.000

The final MAP@3 is the mean of these scores across all questions.

---

Tech Stack

Programming

- Python
- PyTorch
- Pandas
- NumPy

Machine Learning / NLP

- Scikit-learn
- TF-IDF
- Sentence Transformers
- Hugging Face Transformers
- DeBERTa-v3-base
- Cross-Encoder

Retrieval

- FAISS
- Dense embeddings
- Cosine similarity
- LangChain

Experiment Tracking

- Weights & Biases

Environment

The experiments were developed primarily in a Kaggle notebook environment with GPU acceleration used for the neural-network experiments.

---

Repository Structure

Currently, the repository contains the main project notebook:

dlgenai-t22026/
│
└── DL-23f3004345-notebook-t22026.ipynb

The notebook contains the complete experimentation workflow, including:

- Dataset loading
- Exploratory data analysis
- Train/validation split
- Feature engineering
- TF-IDF processing
- PyTorch data pipeline
- Neural-network training
- Model evaluation
- Submission generation
- FAISS retrieval experiments
- DeBERTa experiments
- Cross-encoder experiments
- Model comparison
- Additional DLGenAI milestone experiments

---

Running the Project

The notebook was written for a Kaggle environment because the dataset paths use:

/kaggle/input/competitions/smart-mcq-solver-challenge/

The competition dataset is therefore required to reproduce the experiments.

The notebook installs the required libraries and then loads:

train.csv
test.csv
sample_submission.csv

The main notebook can be opened directly in Kaggle or another Jupyter environment after adapting the dataset paths.

---

Submission

The first model generates a submission containing:

id,prediction

where "prediction" contains the three highest-ranked answer options.

Example:

id,prediction
1,A E D
2,B A C
3,B D E

The Model 1 submission was saved as:

submission_model1.csv

---

Key Takeaways

This project was primarily an exploration of different ways to solve MCQ ranking problems using both classical NLP and modern deep learning.

The experiments explored the progression:

TF-IDF
  ↓
Neural Network
  ↓
Dense Retrieval
  ↓
RAG
  ↓
Pretrained Transformer
  ↓
Cross-Encoder Reranking

One of the most interesting observations was that the more sophisticated architecture did not automatically produce a better competition score.

The simple TF-IDF + MLP model achieved a public MAP@3 of 0.74854, while the DeBERTa retrieval approach achieved 0.74438 and the cross-encoder approach achieved 0.53325.

The project therefore also highlights an important practical ML lesson:

«A more complex model is not necessarily a better model for a particular dataset or evaluation setup.»

---

Author

Sahanawaz Hussain

IIT Madras
BS in Data Science and Applications

---

Note

This repository contains the experimental notebook used during development. Some later experiments and alternative approaches in the notebook are intentionally left commented out as part of the experimentation process.