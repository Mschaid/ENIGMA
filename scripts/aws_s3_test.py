import boto3


class S3Manager:
    def __init__(self):
        pass

    def connect_to_s3(self):
        self.s3_connection = boto3.client('s3')

    def get_available_buckets(self):
        try:
            response = self.s3_connection.list_buckets()
            for bucket in response['Buckets']:
                print(bucket['Name'])
        except Exception as e:
            print(f"Error: {e}")
# main function to call AWSS3Manager


def main():
    s3_manager = S3Manager()
    s3_manager.connect_to_s3()
    s3_manager.get_available_buckets()


# function to upload all files in a directory to s3 bucket


if __name__ == '__main__':
    main()
