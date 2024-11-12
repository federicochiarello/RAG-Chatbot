import pandas as pd
from langchain.schema.document import Document


def csv_loader(directory):
    """
    Processes a list of CSV files, loads data, and returns the aggregated data.

    Returns:
        list: A list containing the aggregated data from all CSV files. Every entry in the list correspond to a row of the CSV
    """
    file_paths = [f for f in directory.iterdir() if f.is_file() and f.suffix == '.csv']
    data = []

    print("----- LOADING CSV FILES -----\n")
    print(f"CSV directory: {str(directory)}")
    print(f"CSV files to be loaded: {len(file_paths)}")

    for file_path in file_paths:
        file_data = parse_csv_to_list(file_path=str(file_path))  # Ensure loader receives a string path
        data.extend(file_data)

    print("CSVs loading completed.")

    return data


def parse_csv_to_list(file_path):
    """
    Processes the content of a CSV file, loads data, and create a list containing one element per row. 
    Each element of the list corresponds to a string. The string content is structured as follows:

    "
    col_1: df[x][1]
    ...
    col_n: df[x][n]
    "

    If the content of a cell is missing, then the corresponding row in the string is removed.

    Returns:
        list: A list containing the content of the CSV file. Every entry in the list correspond to a row of the CSV
    """
    df = pd.read_csv(file_path)
    formatted_rows = []
    
    for index, row in df.iterrows():
        
        row_content = []
        for col in df.columns:
            cell_content = row[col]
            # Only include cells that are not empty
            if pd.notna(cell_content):
                row_content.append(f"{col}: {cell_content}")
        
        # Join each column's content with a newline and add to the list
        document = Document(
            page_content="\n".join(row_content),
            metadata={'source': file_path, 'row': index}
        )
        formatted_rows.append(document)
    
    return formatted_rows