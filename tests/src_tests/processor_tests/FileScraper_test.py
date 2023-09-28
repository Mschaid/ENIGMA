import unittest
from unittest.mock import patch
import os
from src.data_processing.processors.FileScraper import FileScraper


class TestFileScraper(unittest.TestCase):

    def setUp(self):
        self.test_directory = "/path/to/test_directory"
        self.file_scraper = FileScraper()

    def test_init_without_data_directory(self):
        # Act
        scraper = FileScraper()

        # Assert
        self.assertIsNone(scraper._directory)
        self.assertIsNone(scraper._file_names)

    def test_init_with_data_directory(self):
        # Arrange
        data_directory = "/path/to/data"

        # Act
        scraper = FileScraper(data_directory)

        # Assert
        self.assertEqual(scraper._directory, data_directory)
        # self.assertIsNone(scraper._file_names)

    def test_set_directory(self):
        # Act
        self.file_scraper.set_directory(self.test_directory)

        # Assert
        self.assertEqual(self.file_scraper._directory, self.test_directory)

    def test_fetch_all_file_names(self):
        # Arrange
        expected_file_names = [
            "/path/to/test_directory/file1.txt",
            "/path/to/test_directory/file2.txt",
            "/path/to/test_directory/subdirectory/file3.txt"
        ]

        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [
                ("/path/to/test_directory", [], ["file1.txt", "file2.txt"]),
                ("/path/to/test_directory/subdirectory", [], ["file3.txt"])
            ]

            # Act
            self.file_scraper.set_directory(self.test_directory)
            self.file_scraper.fetch_all_file_names()

            # Assert
            self.assertEqual(self.file_scraper._file_names,
                             expected_file_names)

    def test_filter_files_by_extention(self):
        # Arrange
        self.file_scraper._file_names = [
            "/path/to/test_directory/file1.txt",
            "/path/to/test_directory/file2.csv",
            "/path/to/test_directory/file3.txt"
        ]
        extensions = (".csv",)

        # Act
        self.file_scraper.filter_files_by_extention(*extensions)

        # Assert
        expected_file_names = [
            "/path/to/test_directory/file2.csv"
        ]
        self.assertEqual(self.file_scraper._file_names, expected_file_names)

    def test_filter_files_by_extention_no_files_found(self):
        # Arrange
        assumed_files = [
            "/path/to/test_directory/file1.txt",
            "/path/to/test_directory/file2.csv",
            "/path/to/test_directory/file3.txt"
        ]
        self.file_scraper._file_names = assumed_files
        none_extensions = (".png")

        # Act
        self.file_scraper.filter_files_by_extention(*none_extensions)

        # Assert
        self.assertEqual(self.file_scraper._file_names, assumed_files)

    def test_filter_files_by_keywords(self):
        # Arrange
        self.file_scraper._file_names = [
            "/path/to/test_directory/file1.txt",
            "/path/to/test_directory/file2.csv",
            "/path/to/test_directory/file3.txt"
        ]
        keywords = ("file1", "file3")

        # Act
        self.file_scraper.filter_files_by_keywords(*keywords)

        # Assert
        expected_file_names = [
            "/path/to/test_directory/file1.txt",
            "/path/to/test_directory/file3.txt"
        ]
        self.assertEqual(self.file_scraper._file_names, expected_file_names)

    def test_filter_files_by_keywords_no_files_found(self):
        # Arrange
        assumed_files = [
            "/path/to/test_directory/file1.txt",
            "/path/to/test_directory/file2.csv",
            "/path/to/test_directory/file3.txt"
        ]
        self.file_scraper._file_names = assumed_files
        keywords = ("file4",)

        # Act
        self.file_scraper.filter_files_by_keywords(*keywords)

        # Assert
        self.assertEqual(self.file_scraper._file_names, assumed_files)


if __name__ == "__main__":
    unittest.main()
