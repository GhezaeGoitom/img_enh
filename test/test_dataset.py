import great_expectations as ge
from pathlib import Path

orginalPath = str(Path(Path(__file__).parent.absolute()).parent.absolute())
df = ge.read_csv(f"{orginalPath}/data/raw/image_metadata.csv")

def orderCheck():
    df.expect_table_columns_to_match_ordered_list(
    column_list=["id", "Xshape", "Yshape",
    "File_Extension"])



def checkIfNotNull():
    df.expect_column_values_to_not_be_null('Xshape')
    df.expect_column_values_to_not_be_null('Yshape')
    df.expect_column_values_to_not_be_null('File_Extension')


def rangeCheck():
    df.expect_column_values_to_be_between('Xshape',300,301)
    df.expect_column_values_to_be_between('Yshape',300,600)



# if __name__ == "__main__":
    checkIfNotNull()
