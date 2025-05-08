import json, glob, sys
import matplotlib.pyplot as plt

def load_scores(pattern="results*.json"):
    """Return list of (model_name, accuracy_percent)."""
    records = [("Johno", 71)]
    for path in glob.glob(pattern):
        with open(path) as f:
            data = json.load(f)

        # Prefer the explicit 'accuracy' field; derive if missing.
        if "accuracy" in data and data["accuracy"] is not None:
            acc = data["accuracy"]
        elif "correct" in data and "total" in data and data["total"]:
            acc = data["correct"] / data["total"]
        else:
            print(f"⚠️  skipped {path} (no accuracy)", file=sys.stderr)
            continue

        records.append((data["model"], acc * 100))   # convert to %
    return records

def plot(scores, outfile="results.png"):
    # Order by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*scores)

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(labels) + 2))
    y_pos = range(len(labels))

    ax.barh(y_pos, values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()                     # best at top
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(20, 75)                   # as requested
    ax.set_title("SpecID MCQ Accuracy across LLMs")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # annotate bars
    for v, y in zip(values, y_pos):
        ax.text(v + 1, y, f"{int(v)} %", va="center")

    plt.tight_layout()
    fig.savefig(outfile, dpi=180)
    print(f"✅  saved {outfile}")

if __name__ == "__main__":
    data = load_scores()
    if not data:
        sys.exit("No results*.json files found.")
    plot(data)
