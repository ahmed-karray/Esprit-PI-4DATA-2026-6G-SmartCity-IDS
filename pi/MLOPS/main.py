import argparse
from model_pipeline import (
    list_dataset_files,
    prepare_data,
    train_model,
    evaluate_model,
)


def main():
    parser = argparse.ArgumentParser(description="6G IDS Notebook-Faithful LightGBM runner")

    parser.add_argument("--list-data", action="store_true", help="List datasets in ../Data5G")
    parser.add_argument("--prepare", action="store_true", help="Prepare one dataset exactly like notebook")
    parser.add_argument("--train", action="store_true", help="Train LightGBM on one dataset exactly like notebook")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model on one dataset")
    parser.add_argument("--all", action="store_true", help="Train LightGBM on all notebook datasets")

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name: eMBB, mMTC, URLLC, TON_IoT, train_test_network"
    )

    args = parser.parse_args()

    if args.list_data:
        files = list_dataset_files()
        print("Datasets found:")
        for k, v in files.items():
            print(f" - {k}: {v}")
        return

    if args.prepare or args.train or args.evaluate:
        if not args.dataset:
            raise SystemExit("Error: --dataset is required with --prepare, --train, and --evaluate")

    if args.prepare:
        info = prepare_data(dataset_name=args.dataset)
        print("[OK] Data preparation completed.")
        print(f"[OK] Notebook dataset name: {info['dataset_name']}")
        print(f"[OK] Using {len(info['features'])} features")

    elif args.train:
        result = train_model(dataset_name=args.dataset)
        print("\n[OK] Model: LightGBM")
        print(f"[OK] Dataset: {result['dataset_name']}")
        print(f"[OK] Accuracy: {result['accuracy']:.4f}")
        print(f"[OK] F1 macro: {result['f1_macro']:.4f}")
        print(f"[OK] ROC-AUC: {result['roc_auc']:.4f}")

    elif args.evaluate:
        result = evaluate_model(dataset_name=args.dataset)
        print("\n[OK] Evaluation finished.")
        print(f"[OK] Dataset: {result['dataset_name']}")
        print(f"[OK] Accuracy: {result['accuracy']:.4f}")
        print(f"[OK] F1 macro: {result['f1_macro']:.4f}")
        print(f"[OK] ROC-AUC: {result['roc_auc']:.4f}")

    elif args.all:
        datasets = ["mMTC", "URLLC", "eMBB", "TON_IoT"]
        results = []

        print("\n==============================")
        print(" RUNNING ALL DATASETS ")
        print("==============================")

        for ds in datasets:
            try:
                result = train_model(dataset_name=ds)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] {ds} failed: {e}")

        print("\n==============================")
        print(" FINAL RESULTS ")
        print("==============================")
        for r in results:
            print(
                f"{r['dataset_name']:20} | Model: LightGBM   | "
                f"Accuracy: {r['accuracy']:.6f} | "
                f"F1_macro: {r['f1_macro']:.6f} | "
                f"ROC_AUC: {r['roc_auc']:.6f}"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()