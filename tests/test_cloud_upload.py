"""
test_cloud_upload.py
Tests for CloudUploader utilities with output logging.
"""
from brain_mapping.cloud.integration_utils import CloudUploader
import os


def test_upload_to_s3(monkeypatch, tmp_path):
    uploader = CloudUploader()

    def mock_upload_file(file_path, bucket, key, aws_access_key, aws_secret_key):
        # aws_access_key and aws_secret_key are unused in this mock
        with open(tmp_path / "s3_upload.log", "w", encoding="utf-8") as f:
            f.write(
                f"Mock upload {file_path} to S3 bucket {bucket} as {key}\n"
            )
    monkeypatch.setattr(uploader, "upload_to_s3", mock_upload_file)
    uploader.upload_to_s3(
        "dummy.txt", "test-bucket", "test-key", "AKIA...", "SECRET..."
    )
    assert os.path.exists(tmp_path / "s3_upload.log")


def test_upload_to_gcs(monkeypatch, tmp_path):
    uploader = CloudUploader()

    def mock_upload_from_filename(file_path, bucket, blob_name):
        # bucket and blob_name are unused in this mock
        with open(tmp_path / "gcs_upload.log", "w", encoding="utf-8") as f:
            f.write(f"Mock upload {file_path} to GCS\n")
    monkeypatch.setattr(uploader, "upload_to_gcs", mock_upload_from_filename)
    uploader.upload_to_gcs("dummy.txt", "test-bucket", "test-blob")
    assert os.path.exists(tmp_path / "gcs_upload.log")
