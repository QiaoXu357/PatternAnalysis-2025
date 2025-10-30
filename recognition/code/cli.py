import argparse
import os
import torch

from adni.dataset import prepare_data, create_data_loaders
from adni.train import train as train_run
from adni.predict import evaluate_checkpoint, predict_image


def parse_args():
    """Parse command-line arguments for training and evaluation."""
    parser = argparse.ArgumentParser(description="AD vs NC classifier CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    p_train = subparsers.add_parser("train", help="Train model")
    p_train.add_argument("--data-dir", default="AD_NC", help="Dataset root containing train/ and test/")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.add_argument("--weight-decay", type=float, default=0.01)
    p_train.add_argument("--label-smoothing", type=float, default=0.1)
    p_train.add_argument("--warmup-epochs", type=int, default=5)
    p_train.add_argument("--ema-decay", type=float, default=0.999)
    p_train.add_argument("--tta", type=int, default=2, help="TTA passes for test (2=orig+flip)")
    p_train.add_argument("--drop-path-rate", type=float, default=0.2)
    p_train.add_argument("--dims", type=int, nargs=4, default=[96, 192, 384, 768])
    p_train.add_argument("--depths", type=int, nargs=4, default=[3, 3, 9, 3])
    p_train.add_argument("--output-dir", default=".")
    p_train.add_argument("--weighted-sampler", choices=["auto","on","off"], default="auto")

    # Eval subcommand
    p_eval = subparsers.add_parser("eval", help="Evaluate checkpoint")
    p_eval.add_argument("--data-dir", default="AD_NC")
    p_eval.add_argument("--batch-size", type=int, default=32)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--tta", type=int, default=2)
    p_eval.add_argument("--label-smoothing", type=float, default=0.1)
    p_eval.add_argument("--drop-path-rate", type=float, default=0.2)
    p_eval.add_argument("--dims", type=int, nargs=4, default=[96, 192, 384, 768])
    p_eval.add_argument("--depths", type=int, nargs=4, default=[3, 3, 9, 3])

    # Predict subcommand (single image)
    p_pred = subparsers.add_parser("predict", help="Predict a single image")
    p_pred.add_argument("--image", required=True, help="Path to the input image")
    p_pred.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    p_pred.add_argument("--tta", type=int, default=2, help="TTA passes (2=orig+flip)")
    p_pred.add_argument("--drop-path-rate", type=float, default=0.2)
    p_pred.add_argument("--dims", type=int, nargs=4, default=[96, 192, 384, 768])
    p_pred.add_argument("--depths", type=int, nargs=4, default=[3, 3, 9, 3])

    return parser.parse_args()


def main():
    """Entry point for the CLI; dispatches to train/eval routines."""
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.command == "train":
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(args.data_dir)
        loaders = create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=args.batch_size,
            use_weighted_sampler=args.weighted_sampler,
        )
        result = train_run(
            data_loaders=loaders,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            ema_decay=args.ema_decay,
            tta=args.tta,
            model_kwargs={
                'in_chans': 3,
                'num_classes': 2,
                'depths': args.depths,
                'dims': args.dims,
                'drop_path_rate': args.drop_path_rate,
            },
            device=device,
            output_dir=args.output_dir,
        )
        print(f"Best checkpoint: {result['best_checkpoint']}")

    elif args.command == "eval":
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(args.data_dir)
        _, _, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=args.batch_size,
        )
        metrics = evaluate_checkpoint(
            data_loader=test_loader,
            checkpoint_path=args.checkpoint,
            model_kwargs={
                'in_chans': 3,
                'num_classes': 2,
                'depths': args.depths,
                'dims': args.dims,
                'drop_path_rate': args.drop_path_rate,
            },
            device=device,
            tta=args.tta,
            label_smoothing=args.label_smoothing,
        )
        print(f"Accuracy: {metrics['acc']:.4f}, Loss: {metrics['loss']:.4f}")
        print(metrics['report'])

    elif args.command == "predict":
        result = predict_image(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            model_kwargs={
                'in_chans': 3,
                'num_classes': 2,
                'depths': args.depths,
                'dims': args.dims,
                'drop_path_rate': args.drop_path_rate,
            },
            device=device,
            tta=args.tta,
        )
        print(f"Predicted: {result['label']} (index={result['index']})")
        print("Probabilities:")
        for k, v in result['probs'].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
