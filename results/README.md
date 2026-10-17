# Saved results

We tested whether expected speech sounds help predict Whisper's word errors, then checked whether its own confidence score was more useful.

Phoneme proportions and within-word sound pairs did not improve on lexical controls in development. Stress features showed a nominal Base squared-error improvement on new speakers, but about 98% of that gain came from one compound-word clip, `5280-204997-0049`. We have not established a reliable sound effect.

The CMUdict word audit is descriptive: among 726 clips whose reference tokens survived normalization unchanged, words missing from the dictionary were wrong more often. Dictionary absence does not establish what Whisper encountered during training.

For confidence, we fitted on the old 3,000 clips from 112 speakers and froze the predictors before transcribing 1,000 clips from 40 new speakers. All 185 previously used speakers were excluded.

MAE measures how far predicted WER is from measured WER, in percentage points:

| Predictor | Confirmation MAE |
| --- | ---: |
| Training average | 7.20 |
| Training median | 6.00 |
| Recognized-text controls | 6.02 |
| Controls + confidence | 5.15 |

Confidence improved MAE by 14.5% over text alone and helped 34 of 40 speakers. The paired speaker-bootstrap 95% interval for the reduction was 0.63–1.11 points. This supports the predictor on this sample; it does not improve Whisper's transcripts.

Files:

- `phonetic/development/`: 1,000 clips, reference-derived features, folds, three models' transcripts/predictions, and the word audit.
- `phonetic/confirmation/`: 2,000 clips from 80 new speakers, features, transcripts, predictions, and summaries.
- `confidence/`: the 3,000 development clips, separate 1,000-clip confirmation, metadata, predictions, and `summary.json`.
- `manifest.json`: file hashes, row counts, dataset revision, and historical freeze fingerprints.

WERs and WER predictions are fractions in the files (`0.10` = 10%); values above `1` are retained. `phoneme_proportions` vectors follow `phoneme_order` in `manifest.json`.

The phonetic and confidence development Base runs differ on two transcripts and two WER values. Keep each study's targets separate.

Phonetic features use the correct reference transcripts. The confidence comparison uses recognized text and `avg_logprob`; references supply the measured WER.

From the repository root, these checks use CPU and saved data:

```powershell
python -m pip install -r requirements-analysis.txt
python evaluate_confidence.py results/confidence/development_1000.csv results/confidence/development_2000.csv --output outputs/confidence_recheck.csv
python study_evaluation.py develop results/phonetic/development/features.jsonl `
  --results base.en=results/phonetic/development/base.en_results.csv --folds results/phonetic/development/folds.csv `
  --allow-historical-base --output outputs/phonetic_recheck
```

Use new output names if those already exist. To rescore the saved confidence confirmation, run this Python snippet:

```python
import csv
from evaluate_confidence import prediction_errors

with open("results/confidence/confirmation_predictions.csv", encoding="utf-8", newline="") as source:
    rows = list(csv.DictReader(source))
for row in rows:
    for key in row:
        if key.endswith("_prediction") or key == "actual_wer":
            row[key] = float(row[key])
for name, score in prediction_errors(rows).items():
    print(name, "MAE points:", 100 * score["mae"], "MSE:", score["mse"])
```

These are reproductions of existing results. The confirmation speakers are now seen. Audio, model weights and workstation-specific frozen manifests are omitted. The manifest records the original confidence freeze hashes.

Data: [LibriSpeech](https://www.openslr.org/12/), Panayotov, Chen, Povey and Khudanpur (ICASSP 2015), via Hugging Face revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
