## Description:

Our project will investigate whether the phonetic composition of a spoken sentence can be used to predict the Word Error Rate (WER) of an automatic speech recognition system. We will use labeled speech datasets such as LibriSpeech, which provides audio clips paired with ground-truth transcripts, in addition to our own devised/sources transcripts and associated recorded speeches. Each audio clip will be processed through an ASR model, and the predicted transcript will be compared with the correct transcript to calculate WER.

The ground-truth transcripts will then be converted into phoneme sequences using the CMU Pronouncing Dictionary. From these sequences, we will extract sentence-level features such as phoneme frequencies, vowel and consonant proportions, average phonemes per word, consonant clusters, and selected phoneme combinations.

We will first perform exploratory analysis to identify relationships between phonetic features and transcription accuracy. We will then train regression models to predict WER from these features and evaluate how much of the variation in ASR performance can be explained by the phonetic composition of the sentence. If time permits, we will compare multiple ASR models to determine whether similar phonetic patterns lead to errors across different systems.


## Rough Timeline
### 🔵 Mid October
- Finalize the dataset and ASR model.
- Build the initial data-processing pipeline.
- Generate ASR transcripts and calculate WER for each audio clip.
- Convert ground-truth transcripts into phoneme sequences using CMUdict.
- Complete phoneme-based feature engineering.
- Perform exploratory data analysis and create visualizations showing relationships between phonetic features and WER.
### 🔵 Mid November
- Train baseline and more advanced regression models.
- Begin evaluating model performance and identifying important features.
- Finalize model evaluation and interpretation.
### 🔵 End of Semester
- Compare results across models or ASR systems if time permits.
- Complete final visualizations and summarize findings and limitations.
- Clean the GitHub repository and prepare the final report and presentation.

## Goals:

  Automatic speech recognition systems can perform very differently depending on the words and sounds contained in an audio sample. Our project will investigate whether the phonetic composition of a spoken sentence can be used to predict how accurately an automatic speech recognition (ASR) model will transcribe it.

Our main research question is:

Can we predict the Word Error Rate (WER) of an audio clip using features derived from the phoneme composition of its ground-truth transcript?  Success means our model beats a baseline model that predicts the average WER for all training clips in the dataset, measured by mean squared error (MSE) or mean absolute error (MAE).

Rather than predicting whether each individual word is transcribed correctly, we will treat each audio clip or sentence as one observation. For every clip, we will calculate the actual WER produced by an ASR system and extract phoneme-based features from the correct transcript. We will then train regression models to predict WER from these features.

A secondary goal will be to determine whether certain phonemes or phoneme combinations are associated with higher transcription error rates.


## Data Collection:

 We plan to use an existing labeled speech dataset such as LibriSpeech, which contains English audio recordings paired with verified ground-truth transcripts. Using an existing speech corpus will allow us to analyze a large and reproducible set of recordings without needing to collect our own audio.

For each audio clip, we will:
- Obtain the original audio and its ground-truth transcript.
- Process the audio using one or more automatic speech recognition models.
- Compare the predicted transcript with the ground-truth transcript.
- Calculate the Word Error Rate (WER) for the clip.
- Convert the ground-truth transcript into phoneme sequences using the CMU Pronouncing Dictionary (CMUdict).

Word Error Rate will be calculated as:

$$
WER = \frac{S + D + I}{N}
$$

where (S) represents substitutions, (D) represents deletions, (I) represents insertions, and (N) is the number of words in the reference transcript.

We will get and process this data using the Hugging Face datasets library, which provides a convenient interface for downloading and working with speech datasets.


## Feature Extraction

Each audio clip will be represented using features describing the phonetic composition of its transcript.

### Possible features include:

- Frequency of each phoneme
- Proportion of vowels and consonants
- Total number of phonemes
- Average number of phonemes per word
- Number of syllables
- Frequency of consonant clusters
- Frequency of selected phoneme combinations or bigrams
- Sentence length
- Average word length

For example, a sentence containing many occurrences of phonemes such as TH, R, or difficult consonant clusters may have a different transcription error rate from a sentence composed primarily of simpler sound combinations.

## Visualization

Before training models, we will examine how transcription errors vary with different phonetic characteristics.

### Potential visualizations include:

- Distribution of WER across audio clips
- Average WER for sentences containing different phonemes
- Correlation between phoneme frequency and WER
- Heatmaps showing WER associated with different phoneme combinations
- Relationship between sentence length and WER
- Comparison of phoneme-related error patterns across ASR models


## Resources

- [LibriSpeech ASR Corpus](https://www.openslr.org/12/) – Speech dataset containing audio recordings paired with ground-truth transcripts.
- [CMU Pronouncing Dictionary (CMUdict)](https://github.com/cmusphinx/cmudict) – English pronunciation dictionary used to convert words into phoneme sequences.
- [JiWER](https://github.com/jitsi/jiwer) – Python package for calculating Word Error Rate (WER) and other ASR evaluation metrics.
- [OpenAI Whisper](https://github.com/openai/whisper) – Automatic speech recognition model that can be used to generate predicted transcripts.
- [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets) – Large crowdsourced speech dataset that could be used as an extension for testing more diverse speakers and accents.
