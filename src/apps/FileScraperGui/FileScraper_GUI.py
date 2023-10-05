from flask import Flask, render_template, request, jsonify
import os
from src.data_processing.aws_sdk.S3Manager import S3Manager
from src.data_processing.processors.FileScraper import FileScraper

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables
    directory = ''
    file_extensions = ''
    keywords = ''
    results = 'show me'

    # Initialize S3Manager
    file_scraper = FileScraper()

    if request.method == "POST":
        DIRECTORY = request.form.get('directory') or ''
        FILE_EXTENSIONS = request.form.get('file_extensions') or ''
        KEYWORDS = request.form.get('keywords') or None

        file_scraper.scrape_directoy(directory=DIRECTORY,
                                     file_extensions=FILE_EXTENSIONS,
                                     keywords=KEYWORDS)
        results = file_scraper.file_names
        file_scraper.fetch_all_file_names()

        html_content = f"""{results}</p>"""
        return jsonify({"html_content": html_content})
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
