import boto3
from botocore.client import Config

def create_buckets():
    s3 = boto3.resource('s3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='rustfsadmin',
        aws_secret_access_key='rustfsadmin',
        config=Config(signature_version='s3v4'),
        region_name='us-east-1')

    buckets = ['pcb-defect-data', 'pcb-experiment-logs', 'pcb-production-models', 'pcb-test-bucket', 'geos-scratch']
    
    for bucket_name in buckets:
        try:
            s3.create_bucket(Bucket=bucket_name)
            print(f"Created bucket: {bucket_name}")
        except Exception as e:
            print(f"Bucket {bucket_name} might already exist or error: {e}")

if __name__ == "__main__":
    create_buckets()
