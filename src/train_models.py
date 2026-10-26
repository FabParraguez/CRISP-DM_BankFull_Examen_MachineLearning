from __future__ import annotations

import argparse

from src.modeling import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the two CRISP-DM iterations and save model artifacts."
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    result = train_and_save(
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    print("Validation metrics:")
    print(result.validation_metrics.round(4).to_string())
    print("\nGeneralization metrics:")
    print(result.test_metrics.round(4).to_string())
    print(f"\nBest refined model: {result.best_model_name}")
    print("\nHyperparameter search:")
    print(result.search_results.to_string())


if __name__ == "__main__":
    main()
