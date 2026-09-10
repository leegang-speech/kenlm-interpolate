from collections import defaultdict
import math 

def parse_arpa(arpa_path):
    """
    Parse an ARPA file and return n-grams with their probabilities and backoff weights.
    Args:
        arpa_path (str): Path to the ARPA file.
    Returns:
        dict: A dictionary with n-grams as keys and (probability, backoff weight) as values.
    """
    ngrams = {}
    with open(arpa_path, "r") as f:
        in_ngrams_section = False
        current_order = 0

        for line in f:
            line = line.strip()
            if not line or line.startswith("\\end\\"):
                in_ngrams_section = False
                continue

            if line.startswith("\\"):
                if line.endswith("-grams:"):
                    in_ngrams_section = True
                    current_order = int(line.split("-")[0][1:])
                else:
                    in_ngrams_section = False
                continue

            if in_ngrams_section:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                prob = float(parts[0])
                ngram = parts[1]
                backoff = float(parts[2]) if len(parts) > 2 else None
                ngrams[ngram] = (prob, backoff)

    return ngrams


def interpolate_arpa(model1_path, model2_path, output_path, weight1=0.5, weight2=0.5):
    """
    Interpolate two ARPA models and save the resulting ARPA model.
    """
    assert math.isclose(weight1 + weight2, 1.0, rel_tol=1e-9), "Weights must sum to 1."

    # Parse both ARPA models
    ngrams1 = parse_arpa(model1_path)
    ngrams2 = parse_arpa(model2_path)

    # Combine n-grams and interpolate probabilities
    combined_ngrams = defaultdict(lambda: {"prob": [], "backoff": []})
    for ngram, (prob, backoff) in ngrams1.items():
        combined_ngrams[ngram]["prob"].append((prob, weight1))
        if backoff is not None:
            combined_ngrams[ngram]["backoff"].append((backoff, weight1))

    for ngram, (prob, backoff) in ngrams2.items():
        combined_ngrams[ngram]["prob"].append((prob, weight2))
        if backoff is not None:
            combined_ngrams[ngram]["backoff"].append((backoff, weight2))

    interpolated_ngrams = {}
    for ngram, values in combined_ngrams.items():
        interpolated_prob = sum(prob * weight for prob, weight in values["prob"])
        backoff_weights = [b for b in values["backoff"] if b is not None]
        interpolated_backoff = sum(backoff * weight for backoff, weight in backoff_weights) if backoff_weights else None
        interpolated_ngrams[ngram] = (interpolated_prob, interpolated_backoff)

    # Write the interpolated ARPA model
    with open(output_path, "w") as f:
        f.write("\\data\\\n")
        ngram_counts = defaultdict(int)
        for ngram in interpolated_ngrams.keys():
            ngram_counts[len(ngram.split())] += 1

        for n in sorted(ngram_counts.keys()):
            f.write(f"ngram {n}={ngram_counts[n]}\n")

        f.write("\n")
        for n in sorted(ngram_counts.keys()):
            f.write(f"\\{n}-grams:\n")
            for ngram, (prob, backoff) in interpolated_ngrams.items():
                if len(ngram.split()) == n:
                    f.write(f"{prob:.6f}\t{ngram}")
                    if backoff is not None:
                        f.write(f"\t{backoff:.6f}")
                    f.write("\n")
            f.write("\n")

        f.write("\\end\\\n")


interpolate_arpa("ngram_exp/4gram.arpa", "ngram_exp/small.arpa", "ngram_exp/finance4_general6.arpa", weight1=0.6, weight2=0.4)
