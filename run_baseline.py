import csv
import io
import re
from itertools import islice
from pathlib import Path
from shutil import copyfile


# ========== EXPERIMENT SETTINGS ==========

DATASET_NAME = "dev-other"
MODEL_NAME = "base.en"
NUM_CLIPS = 50

# Available LibriSpeech datasets
DATASET_OPTIONS = {
    "dev-clean": ("clean", "validation"),
    "dev-other": ("other", "validation"),
    "test-clean": ("clean", "test"),
    "test-other": ("other", "test")
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULT_COLUMNS = [
    "audio_id",
    "speaker_id",
    "reference_text",
    "predicted_text",
    "wer"
]


def normalize(text, english_normalizer):
    # LibriSpeech has no hyphens ("RELISTED"), so join Whisper's "re-listed"
    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", text)
    return english_normalizer(text)


def extract_phonemes(text, cmu):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

    phonemes = []
    unknown_words = []

    for word in words:
        if word in cmu:
            pronunciation = cmu[word][0]

            for phone in pronunciation:
                phone = re.sub(r"\d", "", phone)
                phonemes.append(phone)
        else:
            unknown_words.append(word)

    return phonemes, unknown_words


def read_results(filename):
    if not filename.exists() or filename.stat().st_size == 0:
        return []

    with filename.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(
                f"Incompatible CSV header in {filename}. "
                f"Expected columns: {', '.join(RESULT_COLUMNS)}"
            )

        results = list(reader)

    for row in results:
        row["wer"] = float(row["wer"])

    return results


def main():
    import cmudict
    import pandas as pd
    import soundfile as sf
    import whisper
    from whisper.normalizers import EnglishTextNormalizer
    from datasets import load_dataset, Audio
    from jiwer import wer

    if DATASET_NAME not in DATASET_OPTIONS:
        raise ValueError(f"Invalid dataset: {DATASET_NAME}")

    english_normalizer = EnglishTextNormalizer()

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_file = OUTPUT_DIR / f"{DATASET_NAME}_results_{MODEL_NAME}.csv"
    phoneme_file = OUTPUT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_phonemes.csv"
    results = read_results(results_file)

    # Reuse Eshaan's uploaded results only for the original experiment.
    uploaded_file = BASE_DIR / "dev_other_results_base.en.csv"
    if (
        not results
        and DATASET_NAME == "dev-other"
        and MODEL_NAME == "base.en"
        and uploaded_file.is_file()
    ):
        results = read_results(uploaded_file)
        copyfile(uploaded_file, results_file)
        print(f"Reusing uploaded results from {uploaded_file.name}")

    completed_ids = {str(row["audio_id"]) for row in results}

    config, split = DATASET_OPTIONS[DATASET_NAME]
    dataset = load_dataset(
        "openslr/librispeech_asr",
        config,
        split=split,
        streaming=True
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    print(f"Using dataset: {DATASET_NAME}")

    model = None
    failed_clips = 0
    write_header = not results_file.exists() or results_file.stat().st_size == 0

    with results_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
            file.flush()

        for i, sample in enumerate(islice(dataset, NUM_CLIPS)):
            audio_id = str(sample["id"])

            if audio_id in completed_ids:
                print(f"Clip {i+1}/{NUM_CLIPS} | {audio_id} | already saved")
                continue

            if model is None:
                print(f"Loading Whisper: {MODEL_NAME}")
                model = whisper.load_model(MODEL_NAME)

            try:
                reference = sample["text"]
                audio_info = sample["audio"]
                audio_source = (
                    io.BytesIO(audio_info["bytes"])
                    if audio_info["bytes"] is not None
                    else audio_info["path"]
                )

                audio, sample_rate = sf.read(audio_source, dtype="float32")

                if audio.ndim == 2:
                    audio = audio.mean(axis=1)

                if sample_rate != 16000:
                    raise ValueError("Expected 16 kHz audio")

                prediction = model.transcribe(
                    audio,
                    language="en",
                    fp16=False
                )["text"]

                error = wer(
                    normalize(reference, english_normalizer),
                    normalize(prediction, english_normalizer)
                )

                row = {
                    "audio_id": audio_id,
                    "speaker_id": sample["speaker_id"],
                    "reference_text": reference,
                    "predicted_text": prediction,
                    "wer": error
                }

            except Exception as clip_error:
                failed_clips += 1
                print(
                    f"Clip {i+1}/{NUM_CLIPS} | {audio_id} | failed: {clip_error}"
                )
                continue

            # Keep disk errors outside the per-clip error handler.
            writer.writerow(row)
            file.flush()
            results.append(row)
            completed_ids.add(audio_id)

            print(
                f"Clip {i+1}/{NUM_CLIPS} | "
                f"WER: {row['wer']:.2%}"
            )

    print(f"\nTranscriptions saved to {results_file}")
    print("Failed clips:", failed_clips)

    if not results:
        print("No successful transcriptions to process.")
        return 1 if failed_clips else 0

    df = pd.DataFrame(results, columns=RESULT_COLUMNS)

    print("\n--- SUMMARY ---")
    print("Total saved clips:", len(df))
    print("Average clip WER:", df["wer"].mean())
    print("Clips with errors:", (df["wer"] > 0).sum())

    # Rebuild phonemes from the saved transcriptions.
    cmu = cmudict.dict()
    phoneme_results = df["reference_text"].apply(
        lambda text: extract_phonemes(text, cmu)
    )

    df["phonemes"] = phoneme_results.apply(lambda x: " ".join(x[0]))
    df["unknown_words"] = phoneme_results.apply(lambda x: ", ".join(x[1]))
    df["num_phonemes"] = phoneme_results.apply(lambda x: len(x[0]))

    df.to_csv(phoneme_file, index=False)

    print("\n--- PHONEME EXTRACTION ---")
    print(df[["reference_text", "phonemes", "wer"]].head())
    print("Clips with unknown words:", (df["unknown_words"] != "").sum())
    print(f"Phonemes saved to {phoneme_file}")
    return 1 if failed_clips else 0


if __name__ == "__main__":
    raise SystemExit(main())
