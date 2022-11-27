import boto3
from botocore.exceptions import NoCredentialsError

# S3 Bucket policy modified and role added to 
# ec2 instance so that file can be uploaded 
# directly from program


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

print("Starting to copy last.ckpt ..... ")

local_file='logs/train/runs/2022-11-25_14-20-42/checkpoints/last.ckpt'
bucket_name='jrl-aws-bucket'
s3_file_name='last.ckpt'
uploaded = upload_to_aws(local_file, bucket_name, s3_file_name)

print("File last.ckpt copied ..... ",uploaded)
