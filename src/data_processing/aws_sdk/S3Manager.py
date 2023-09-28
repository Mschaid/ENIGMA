import boto3


class S3Manager:
    def __init__(self):

        self._s3_connection = None

    def connect_to_s3(self):
        """
        Establishes a connection to the Amazon S3 service.

        Initializes the `s3_connection` attribute of the current instance with a client object
        from the boto3 library, which enables communication with the Amazon S3 service. The client
        object is created using the `boto3.client()` method, passing in the string 's3' as the
        service name.

        Returns
        -------
        None
        """

        try:
            self._s3_connection = boto3.client('s3')
        except Exception as e:
            print(f"Error: {e}")

    @property
    def s3_connection(self):
        """
        Get the S3 connection.

        Returns:
            The S3 connection object.
        """
        if self._s3_connection is None:
            print(f"Connection not established: Calling connect_to_s3() now.")
            self.connect_to_s3()
        return self._s3_connection

# @property
    def get_available_buckets(self):
        """
        Retrieves a list of available S3 buckets.

        Prints the name of each bucket.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs while listing the buckets.
        """

        try:
            response = self.s3_connection.list_buckets()
            for bucket in response['Buckets']:
                print(bucket['Name'])
        except Exception as e:
            print(f"Error: {e}")


def main():
    s3_manager = S3Manager()
    s3_manager.connect_to_s3()
    s3_manager.s3_connection

    s3_manager.get_available_buckets()


# function to upload all files in a directory to s3 bucket


if __name__ == '__main__':
    main()
