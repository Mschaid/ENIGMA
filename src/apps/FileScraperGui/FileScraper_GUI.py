from flask import Flask, render_template, request, jsonify
import os
from src.data_processing.aws_sdk.S3Manager import S3Manager
from src.data_processing.processors.FileScraper import FileScraper

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def file_scraper():
    # Initialize variables
    directory = ''
    file_extensions = ''
    keywords = ''
    files_found_results = ''
    filetypes_not_found = ''
    keywords_not_found = ''

    # Initialize S3Manager

    if request.method == "POST":
        DIRECTORY = request.form.get('directory') or None
        FILE_EXTENSIONS = request.form.get('file_extensions') or None
        KEYWORDS = request.form.get('keywords') or None

        file_scraper = FileScraper(DIRECTORY)
        file_scraper.scrape_directoy(file_extensions=FILE_EXTENSIONS,
                                     keywords=KEYWORDS)

        files_found_results = file_scraper.search_results['file_names']
        filetypes_not_found = file_scraper.search_results['extensions_not_found']
        keywords_not_found = file_scraper.search_results['keywords_not_found']

        files_found_content = f"""{files_found_results}</p>"""
        filetypes_not_found_content = f"""{filetypes_not_found}</p>"""
        keywords_not_found_content = f"""{keywords_not_found}</p>"""

        return jsonify({"files_found_content": files_found_content,
                        "filetypes_not_found_content": filetypes_not_found_content,
                        "keywords_not_found_content": keywords_not_found_content})
    return render_template("file_scraper.html")


@app.route('/s3', methods=['GET', 'POST'])
def s3_manager():
    return render_template("s3_manager.html")


if __name__ == "__main__":
    app.run(debug=True)
