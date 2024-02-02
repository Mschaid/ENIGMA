import dataframe_image as dfi


def main():
    dfi.convert("/Users/mds8301/Development/ENIGMA/notebooks/reports/final_experiments_analysis.ipynb",
                to='pdf',
                use='browser',
                center_df=True,
                max_rows=30,
                max_cols=10,
                execute=False,
                save_notebook=False,
                limit=None,
                document_name=None,
                table_conversion='chrome',
                chrome_path=None,
                latex_command=None,
                output_dir=None,
                )


if __name__ == "__main__":
    main()
