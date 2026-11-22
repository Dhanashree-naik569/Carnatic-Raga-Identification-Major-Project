
import os
import glob
from collections import defaultdict

from predict import predict_file, model_available

EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

if not model_available():
    print("ERROR: Saved model could not be loaded.")
    raise SystemExit(1)

folders = sorted(glob.glob(os.path.join("dataset", "*")))

total = 0
correct = 0
failed = 0
results = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})

print("\n========== FULL DATASET TEST ==========\n")

for folder in folders:
    if not os.path.isdir(folder):
        continue

    actual = os.path.basename(folder)

    files = sorted(
        f for f in glob.glob(os.path.join(folder, "*"))
        if os.path.isfile(f)
        and os.path.splitext(f)[1].lower() in EXTENSIONS
    )

    for path in files:
        total += 1
        results[actual]["total"] += 1

        print(f"\nActual: {actual} | File: {os.path.basename(path)}")

        try:
            result = predict_file(path)
            if "error" in result:
                raise RuntimeError(result["error"])

            predicted = result["prediction"]
            confidence = result["confidence"]

            print(f"Predicted: {predicted} | Score: {confidence}%")

            if predicted.casefold() == actual.casefold():
                correct += 1
                results[actual]["correct"] += 1
                print("Result: CORRECT")
            else:
                print("Result: WRONG")

        except Exception as error:
            failed += 1
            results[actual]["failed"] += 1
            print(f"Result: FAILED | {error}")

print("\n========== PER-RAGA RESULTS ==========")
print(f"{'Raga':20} {'Correct':>8} {'Tested':>8} {'Accuracy':>10}")

for raga, data in sorted(results.items()):
    tested = data["total"] - data["failed"]
    accuracy = data["correct"] / tested * 100 if tested else 0
    print(
        f"{raga:20} {data['correct']:>8} "
        f"{tested:>8} {accuracy:>9.2f}%"
    )

successful = total - failed

print("\n========== FINAL RESULTS ==========")
print(f"Total files found : {total}")
print(f"Correct           : {correct}")
print(f"Wrong             : {successful - correct}")
print(f"Failed            : {failed}")

if successful:
    print(f"Accuracy          : {correct / successful * 100:.2f}%")
else:
    print("Accuracy          : N/A")

print("\nNote: This tests the saved ML model on dataset recordings.")
print("It is not an independent test of unseen recordings.")
