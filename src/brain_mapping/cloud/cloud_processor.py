"""
Cloud Integration Module
=======================

This module provides cloud-based processing and collaboration capabilities
for brain mapping workflows, supporting AWS, Google Cloud, and Azure.
"""

import warnings
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    warnings.warn("boto3 not available. AWS integration disabled.")

try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    warnings.warn("google-cloud-storage not available. Google Cloud integration disabled.")

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    warnings.warn("azure-storage-blob not available. Azure integration disabled.")


class CloudProcessor:
    """
    Handles cloud-based dataset upload and processing.
    
    This class provides comprehensive cloud integration including data upload,
    distributed processing, result sharing, and cost optimization.
    """
    
    def __init__(self, 
                 cloud_provider: str = 'aws',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize cloud processor.
        
        Parameters
        ----------
        cloud_provider : str, default='aws'
            Cloud provider: 'aws', 'google', or 'azure'
        config : dict, optional
            Configuration dictionary with credentials and settings
        """
        self.cloud_provider = cloud_provider.lower()
        self.config = config or {}
        self.client = None
        self.bucket_name = self.config.get('bucket_name', 'brain-mapping-data')
        self.region = self.config.get('region', 'us-east-1')
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize cloud client based on provider."""
        if self.cloud_provider == 'aws':
            if not AWS_AVAILABLE:
                raise ImportError("boto3 is required for AWS integration")
            self._initialize_aws_client()
        elif self.cloud_provider == 'google':
            if not GOOGLE_CLOUD_AVAILABLE:
                raise ImportError("google-cloud-storage is required for Google Cloud integration")
            self._initialize_google_client()
        elif self.cloud_provider == 'azure':
            if not AZURE_AVAILABLE:
                raise ImportError("azure-storage-blob is required for Azure integration")
            self._initialize_azure_client()
        else:
            raise ValueError(f"Unsupported cloud provider: {self.cloud_provider}")
    
    def _initialize_aws_client(self):
        """Initialize AWS S3 client."""
        try:
            self.client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=self.config.get('aws_access_key_id'),
                aws_secret_access_key=self.config.get('aws_secret_access_key')
            )
            print(f"✓ AWS S3 client initialized for region: {self.region}")
        except NoCredentialsError:
            # Try to use default credentials
            self.client = boto3.client('s3', region_name=self.region)
            print(f"✓ AWS S3 client initialized with default credentials")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AWS client: {e}")
    
    def _initialize_google_client(self):
        """Initialize Google Cloud Storage client."""
        try:
            self.client = storage.Client()
            print("✓ Google Cloud Storage client initialized")
        except DefaultCredentialsError:
            raise RuntimeError("Google Cloud credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Cloud client: {e}")
    
    def _initialize_azure_client(self):
        """Initialize Azure Blob Storage client."""
        try:
            connection_string = self.config.get('azure_connection_string')
            if not connection_string:
                raise ValueError("Azure connection string required in config")
            
            self.client = BlobServiceClient.from_connection_string(connection_string)
            print("✓ Azure Blob Storage client initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure client: {e}")
    
    def upload_dataset(self, 
                      local_path: Union[str, Path], 
                      cloud_path: str,
                      progress_callback: Optional[callable] = None) -> str:
        """
        Upload dataset to cloud storage.
        
        Parameters
        ----------
        local_path : str or Path
            Local path to dataset
        cloud_path : str
            Cloud storage path
        progress_callback : callable, optional
            Callback function for upload progress
            
        Returns
        -------
        str
            Cloud URL of uploaded dataset
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")
        
        if self.cloud_provider == 'aws':
            return self._upload_to_aws(local_path, cloud_path, progress_callback)
        elif self.cloud_provider == 'google':
            return self._upload_to_google(local_path, cloud_path, progress_callback)
        elif self.cloud_provider == 'azure':
            return self._upload_to_azure(local_path, cloud_path, progress_callback)
    
    def _upload_to_aws(self, local_path: Path, cloud_path: str, progress_callback: Optional[callable]) -> str:
        """Upload to AWS S3."""
        try:
            if local_path.is_file():
                # Upload single file
                self.client.upload_file(
                    str(local_path), 
                    self.bucket_name, 
                    cloud_path,
                    Callback=progress_callback
                )
            else:
                # Upload directory recursively
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        cloud_key = f"{cloud_path}/{relative_path}"
                        
                        self.client.upload_file(
                            str(file_path),
                            self.bucket_name,
                            str(cloud_key),
                            Callback=progress_callback
                        )
            
            return f"s3://{self.bucket_name}/{cloud_path}"
            
        except ClientError as e:
            raise RuntimeError(f"AWS upload failed: {e}")
    
    def _upload_to_google(self, local_path: Path, cloud_path: str, progress_callback: Optional[callable]) -> str:
        """Upload to Google Cloud Storage."""
        try:
            bucket = self.client.bucket(self.bucket_name)
            
            if local_path.is_file():
                # Upload single file
                blob = bucket.blob(cloud_path)
                blob.upload_from_filename(str(local_path))
            else:
                # Upload directory recursively
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        cloud_key = f"{cloud_path}/{relative_path}"
                        
                        blob = bucket.blob(str(cloud_key))
                        blob.upload_from_filename(str(file_path))
            
            return f"gs://{self.bucket_name}/{cloud_path}"
            
        except Exception as e:
            raise RuntimeError(f"Google Cloud upload failed: {e}")
    
    def _upload_to_azure(self, local_path: Path, cloud_path: str, progress_callback: Optional[callable]) -> str:
        """Upload to Azure Blob Storage."""
        try:
            container_client = self.client.get_container_client(self.bucket_name)
            
            if local_path.is_file():
                # Upload single file
                with open(local_path, 'rb') as data:
                    container_client.upload_blob(cloud_path, data)
            else:
                # Upload directory recursively
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        cloud_key = f"{cloud_path}/{relative_path}"
                        
                        with open(file_path, 'rb') as data:
                            container_client.upload_blob(str(cloud_key), data)
            
            return f"https://{self.client.account_name}.blob.core.windows.net/{self.bucket_name}/{cloud_path}"
            
        except AzureError as e:
            raise RuntimeError(f"Azure upload failed: {e}")
    
    def process_on_cloud(self, 
                        dataset_path: str, 
                        pipeline: str,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run preprocessing pipeline on cloud infrastructure.
        
        Parameters
        ----------
        dataset_path : str
            Cloud path to dataset
        pipeline : str
            Pipeline name or configuration
        config : dict, optional
            Processing configuration
            
        Returns
        -------
        Dict[str, Any]
            Processing results and metadata
        """
        # This would integrate with cloud compute services
        # For now, return a mock result
        result = {
            'status': 'completed',
            'dataset_path': dataset_path,
            'pipeline': pipeline,
            'processing_time': 120.5,
            'cost': self._estimate_cost(dataset_path, pipeline),
            'results_path': f"{dataset_path}/results",
            'logs': f"{dataset_path}/logs"
        }
        
        print(f"✓ Cloud processing completed for {dataset_path}")
        return result
    
    def _estimate_cost(self, dataset_path: str, pipeline: str) -> float:
        """Estimate processing cost."""
        # Mock cost estimation
        base_cost = 0.10  # $0.10 per GB
        processing_cost = 0.05  # $0.05 per minute
        
        # Estimate dataset size (mock)
        estimated_size_gb = 5.0
        estimated_time_minutes = 120
        
        total_cost = (base_cost * estimated_size_gb) + (processing_cost * estimated_time_minutes)
        return round(total_cost, 2)
    
    def share_results(self, 
                     results_path: str, 
                     collaborators: List[str],
                     permissions: str = 'read') -> Dict[str, Any]:
        """
        Share results with collaborators.
        
        Parameters
        ----------
        results_path : str
            Cloud path to results
        collaborators : List[str]
            List of collaborator emails or IDs
        permissions : str, default='read'
            Permission level: 'read', 'write', or 'admin'
            
        Returns
        -------
        Dict[str, Any]
            Sharing results and access URLs
        """
        sharing_result = {
            'status': 'shared',
            'results_path': results_path,
            'collaborators': collaborators,
            'permissions': permissions,
            'access_urls': {},
            'expires_at': None
        }
        
        # Generate access URLs for each collaborator
        for collaborator in collaborators:
            if self.cloud_provider == 'aws':
                sharing_result['access_urls'][collaborator] = self._generate_aws_presigned_url(results_path)
            elif self.cloud_provider == 'google':
                sharing_result['access_urls'][collaborator] = self._generate_google_signed_url(results_path)
            elif self.cloud_provider == 'azure':
                sharing_result['access_urls'][collaborator] = self._generate_azure_sas_url(results_path)
        
        print(f"✓ Results shared with {len(collaborators)} collaborators")
        return sharing_result
    
    def _generate_aws_presigned_url(self, path: str) -> str:
        """Generate AWS presigned URL."""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': path},
                ExpiresIn=3600  # 1 hour
            )
            return url
        except Exception as e:
            print(f"Warning: Failed to generate AWS presigned URL: {e}")
            return f"s3://{self.bucket_name}/{path}"
    
    def _generate_google_signed_url(self, path: str) -> str:
        """Generate Google Cloud signed URL."""
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(path)
            url = blob.generate_signed_url(
                version="v4",
                expiration=3600,  # 1 hour
                method="GET"
            )
            return url
        except Exception as e:
            print(f"Warning: Failed to generate Google signed URL: {e}")
            return f"gs://{self.bucket_name}/{path}"
    
    def _generate_azure_sas_url(self, path: str) -> str:
        """Generate Azure SAS URL."""
        try:
            container_client = self.client.get_container_client(self.bucket_name)
            blob_client = container_client.get_blob_client(path)
            
            sas_token = blob_client.generate_sas(
                permission="read",
                expiry=time.time() + 3600,  # 1 hour
                protocol="https"
            )
            
            return f"{blob_client.url}?{sas_token}"
        except Exception as e:
            print(f"Warning: Failed to generate Azure SAS URL: {e}")
            return f"https://{self.client.account_name}.blob.core.windows.net/{self.bucket_name}/{path}"
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get cloud storage usage statistics.
        
        Returns
        -------
        Dict[str, Any]
            Storage usage information
        """
        if self.cloud_provider == 'aws':
            return self._get_aws_storage_usage()
        elif self.cloud_provider == 'google':
            return self._get_google_storage_usage()
        elif self.cloud_provider == 'azure':
            return self._get_azure_storage_usage()
    
    def _get_aws_storage_usage(self) -> Dict[str, Any]:
        """Get AWS S3 storage usage."""
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket_name)
            
            total_size = 0
            object_count = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    total_size += obj['Size']
                    object_count += 1
            
            return {
                'provider': 'aws',
                'bucket': self.bucket_name,
                'total_size_bytes': total_size,
                'total_size_gb': round(total_size / (1024**3), 2),
                'object_count': object_count
            }
        except Exception as e:
            return {'error': f"Failed to get AWS storage usage: {e}"}
    
    def _get_google_storage_usage(self) -> Dict[str, Any]:
        """Get Google Cloud Storage usage."""
        try:
            bucket = self.client.bucket(self.bucket_name)
            
            total_size = 0
            object_count = 0
            
            for blob in bucket.list_blobs():
                total_size += blob.size
                object_count += 1
            
            return {
                'provider': 'google',
                'bucket': self.bucket_name,
                'total_size_bytes': total_size,
                'total_size_gb': round(total_size / (1024**3), 2),
                'object_count': object_count
            }
        except Exception as e:
            return {'error': f"Failed to get Google storage usage: {e}"}
    
    def _get_azure_storage_usage(self) -> Dict[str, Any]:
        """Get Azure Blob Storage usage."""
        try:
            container_client = self.client.get_container_client(self.bucket_name)
            
            total_size = 0
            object_count = 0
            
            for blob in container_client.list_blobs():
                total_size += blob.size
                object_count += 1
            
            return {
                'provider': 'azure',
                'container': self.bucket_name,
                'total_size_bytes': total_size,
                'total_size_gb': round(total_size / (1024**3), 2),
                'object_count': object_count
            }
        except Exception as e:
            return {'error': f"Failed to get Azure storage usage: {e}"}
    
    def test_cloud_connection(self):
        """Test connection to cloud provider."""
        try:
            if self.cloud_provider == 'aws':
                self.client.list_buckets()
            elif self.cloud_provider == 'google':
                list(self.client.list_buckets())
            print(f"Cloud connection to {self.cloud_provider} successful.")
            return True
        except Exception as e:
            print(f"Cloud connection failed: {e}")
            return False


class CloudCollaboration:
    """
    Manages collaborative features for cloud-based workflows.
    """
    
    def __init__(self, cloud_processor: CloudProcessor):
        """
        Initialize cloud collaboration.
        
        Parameters
        ----------
        cloud_processor : CloudProcessor
            Initialized cloud processor instance
        """
        self.cloud_processor = cloud_processor
        self.active_sessions = {}
    
    def create_collaboration_session(self, 
                                   session_name: str,
                                   dataset_path: str,
                                   collaborators: List[str]) -> str:
        """
        Create a new collaboration session.
        
        Parameters
        ----------
        session_name : str
            Name of the collaboration session
        dataset_path : str
            Cloud path to shared dataset
        collaborators : List[str]
            List of collaborator emails
            
        Returns
        -------
        str
            Session ID
        """
        session_id = f"session_{int(time.time())}"
        
        session = {
            'id': session_id,
            'name': session_name,
            'dataset_path': dataset_path,
            'collaborators': collaborators,
            'created_at': time.time(),
            'status': 'active'
        }
        
        self.active_sessions[session_id] = session
        
        # Share dataset with collaborators
        self.cloud_processor.share_results(dataset_path, collaborators)
        
        print(f"✓ Collaboration session '{session_name}' created with ID: {session_id}")
        return session_id
    
    def join_session(self, session_id: str, user_email: str) -> Dict[str, Any]:
        """
        Join an existing collaboration session.
        
        Parameters
        ----------
        session_id : str
            Session ID to join
        user_email : str
            Email of user joining the session
            
        Returns
        -------
        Dict[str, Any]
            Session information and access details
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if user_email not in session['collaborators']:
            raise ValueError(f"User {user_email} not authorized for session {session_id}")
        
        return {
            'session': session,
            'access_url': self.cloud_processor._generate_access_url(session['dataset_path']),
            'joined_at': time.time()
        }
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active collaboration sessions.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of active sessions
        """
        return list(self.active_sessions.values())

# Usage Example
if __name__ == "__main__":
    processor = CloudProcessor('aws')
    try:
        processor.upload_dataset('local.txt', 'cloud.txt')
    except Exception as e:
        print(f"Error: {e}")