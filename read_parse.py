## Read and Parse dataset
import pandas as pd
import os


class GetInput(object):
    """
    Get columns from the data 
    """

    def __init__(self, text: list) -> None:
        """
        get dataframe for data
        input should be given as list of strings

        """
        self.text = text
        self.df = pd.DataFrame()

    def __len__(self) -> int:
        return len(self.text)

    def __get_dataframe__(self) -> None:
        """ Convert to dataframe with index and columns
        """
        self.df = pd.DataFrame(self.text)
        self.df = self.df.reset_index()
        self.df.columns = ['index', 'text']


def main(input_file_path):
    """ Need input file path with text written with enters"""
    if os.path.exists(input_file_path):
        with open(input_file_path, 'r') as file:
            lines = []
            for line in file:
                lines.append(str(line).strip())
        new_obj = GetInput(lines)
        new_obj.__get_dataframe__()
        # write df to output
        new_obj.df.to_csv("./output/temp/input_data.csv", index=False)
    else:
        print("Enter correct path, no file exist")


if __name__ == "__main__":
    main("./input/input.txt")
