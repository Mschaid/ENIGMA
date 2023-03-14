

class FileTypeError(Exception):
    def __init__(self, filetype):
        self.message = f"File type {filetype} is not supported."
        super().__init__(self.message)

    def __str__(self):
        return self.message
