"""
integration_utils.py
Cloud integration utilities for brain mapping toolkit.
"""
import boto3
from google.cloud import storage


class CloudUploader:
    """Upload files to AWS S3 and Google Cloud Storage."""
    def upload_to_s3(
        self, file_path: str, bucket: str, key: str,
        aws_access_key: str, aws_secret_key: str
    ):
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        s3.upload_file(file_path, bucket, key)

    def upload_to_gcs(self, file_path: str, bucket: str, blob_name: str):
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        blob.upload_from_filename(file_path)

    def load_config(self, config_path: str):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
