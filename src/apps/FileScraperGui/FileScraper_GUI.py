from flask import Flask, render_template, request, jsonify
import os
from src.data_processing.aws_sdk.S3Manager import S3Manager

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables
    directory = ''
    file_extensions = ''
    keywords = ''
    results = 'show me'

    # Initialize S3Manager
    s3_manager = S3Manager()

    if request.method == "POST":
        DIRECTORY = request.form.get('directory')
        FILE_EXTENSIONS = request.form.get('file_extensions')
        KEYWORDS = request.form.get('keywords')

        s3_manager.scrape_directoy(directory=DIRECTORY,
                                   file_extensions=FILE_EXTENSIONS,
                                   keywords=KEYWORDS)
        results = s3_manager.scraper.file_names
        s3_manager.scraper.fetch_all_file_names()

        html_content = f"""{results}</p>"""
        return jsonify({"html_content": html_content})
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
