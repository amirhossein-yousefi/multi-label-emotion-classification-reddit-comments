# sagemaker/deploy/train_and_deploy.py
import argparse, os, time
import sagemaker
from sagemaker.huggingface import HuggingFace, HuggingFaceModel

def get_role():
    try:
        return sagemaker.get_execution_role()
    except Exception:
        role = os.environ.get("SAGEMAKER_ROLE_ARN")
        if not role:
            raise RuntimeError("Set SAGEMAKER_ROLE_ARN or run inside SageMaker Studio/Notebook.")
        return role

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--bucket", default=None, help="S3 bucket for training output; defaults to SageMaker default bucket")
    parser.add_argument("--base_job_name", default="reddit-emotions-ml")
    parser.add_argument("--train_s3", default="", help="s3://... prefix for training data (CSV). If empty, script uses --use_hf_goemotions.")
    parser.add_argument("--val_s3", default="", help="s3://... prefix for validation data (CSV)")
    parser.add_argument("--instance_type_train", default="ml.g4dn.xlarge")
    parser.add_argument("--instance_type_infer", default="ml.m5.xlarge")
    # Choose a version triplet supported by the AWS HF DLC matrix
    parser.add_argument("--transformers_version", default="4.28.1")
    parser.add_argument("--pytorch_version", default="1.13")
    parser.add_argument("--py_version", default="py39")
    parser.add_argument("--deploy", action="store_true")

    # training hyperparameters (forwarded to entrypoint/train.py)
    parser.add_argument("--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_hf_goemotions", action="store_true")

    args = parser.parse_args()

    sess = sagemaker.Session()
    role = get_role()
    bucket = args.bucket or sess.default_bucket()

    # source dir contains entrypoints + requirements.txt
    source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    entry_point = "entrypoint/train.py"

    hyperparameters = {
        "model_name_or_path": args.model_name_or_path,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "threshold": args.threshold,
    }
    if args.use_hf_goemotions:
        hyperparameters["use_hf_goemotions"] = True

    estimator = HuggingFace(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        instance_type=args.instance_type_train,
        instance_count=1,
        transformers_version=args.transformers_version,
        pytorch_version=args.pytorch_version,
        py_version=args.py_version,
        base_job_name=args.base_job_name,
        hyperparameters=hyperparameters,
        volume_size=200,  # GB
        disable_profiler=True,
    )

    inputs = None
    if args.train_s3:
        channels = {"train": args.train_s3}
        if args.val_s3:
            channels["validation"] = args.val_s3
        inputs = channels

    estimator.fit(inputs=inputs)
    print("Training complete. Model artifacts:", estimator.model_data)

    if args.deploy:
        model = HuggingFaceModel(
            role=role,
            transformers_version=args.transformers_version,
            pytorch_version=args.pytorch_version,
            py_version=args.py_version,
            model_data=estimator.model_data,
            entry_point="entrypoint/inference.py",
            source_dir=source_dir,
            env={},  # add ENV if needed
        )
        endpoint_name = f"{args.base_job_name}-ep-{int(time.time())}"
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type_infer,
            endpoint_name=endpoint_name,
        )
        print("Deployed endpoint:", endpoint_name)

if __name__ == "__main__":
    main()
