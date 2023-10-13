
import os
from src.data_processing.aws_sdk.S3Manager import S3Manager
from src.data_processing.processors.FileScraper import FileScraper
import streamlit as st


class FileScraperUI:
    def __init__(self):
        self.title = 'FMS File Search'
        self.file_scraper = FileScraper(directory='')
        self.config()

    def config(self):
        st.set_page_config(
            page_title="FMS File Search",
            page_icon=":mag:",
            layout="wide",
            initial_sidebar_state="collapsed",
            menu_items={"About": """
                    # How to use this page:
                    this page is used to search for files in a directory
                    * Simply enter the directory you want to search in the left sidebar
                    * enter the file extensions and keywords you want to search for in the designated feilds
                    * Click search
                    
                    by default, the search will return all files in the directory if no extensions or keywords are provided
                 
                    if no files with extensions provided are found, the search will return nothing 
                     
                    if no keywords are provided, the search will return all files with the provided extensions 

                    """},
        )

    def top_left_search_form_user_input(self):

        st.sidebar.header('Search Files:mag:', divider=True)
        directory = st.sidebar.text_input(
            'Directory to search (enter path):', value='')
        file_extensions = st.sidebar.text_input(
            'File Extensions (comma separated):', value='')
        keywords = st.sidebar.text_input(
            'Keywords (comma seperated):', value='')
        # Button to trigger search
        if st.sidebar.button('Search'):
            # when search is called, set the directory and scrape the files
            self.file_scraper.directory = directory

            self.file_scraper.scrape_directoy(
                file_extensions=file_extensions, keywords=keywords)

            # Display results
            st.header('Search Results', divider="red", anchor='center')
            st.subheader('Files Found')
            # Replace the line below with actual search results

            st.write(
                self.file_scraper.search_results["file_search_results"])

            st.subheader('File Types Not Found')
            # Replace the line below with actual data
            st.write(self.file_scraper.search_results["extensions_not_found"])

            st.subheader('Keywords Not Found')
            # Replace the line below with actual data
            st.write(self.file_scraper.search_results["keywords_not_found"])

# Form

    def main_display(self):
        st.title('FMS File Search')


def main():
    app = FileScraperUI()
    app.main_display()
    app.top_left_search_form_user_input()


if __name__ == '__main__':
    main()
