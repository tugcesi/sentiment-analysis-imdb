# IMDB Sentiment Analysis Project

## Overview
This project aims to perform sentiment analysis on movie reviews from IMDB, determining whether the sentiments expressed in the reviews are positive, negative, or neutral. 

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/tugcesi/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
   ```

2. **Set Up Python Environment**
   It is recommended to use `venv` or `conda` to create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Your Dataset**
   Ensure you have the IMDB dataset of movie reviews. This can be found at [IMDB datasets](https://ai.stanford.edu/~amaas/data/sentiment/).

2. **Run the Sentiment Analysis Model**
   You can run the model using the following command:
   ```bash
   python main.py --input <path_to_your_data>
   ```
   Replace `<path_to_your_data>` with the path to your dataset.

## Model Information
The sentiment analysis model used in this project is based on deep learning techniques (e.g., LSTM, CNN, etc.) and has been trained on the IMDB reviews dataset. The model's architecture consists of:
- Embedding Layer
- LSTM Layer(s)
- Fully Connected Layer

## Contributions
Feel free to fork the repository and submit pull requests to contribute to the project. 

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any inquiries, please reach out to [your_email@example.com].