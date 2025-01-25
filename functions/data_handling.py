import os 
import pandas as pd


# def align_datasets_by_index(df1, df2):
#     """
#     Aligns two DataFrames by their indices (row names), ensuring rows appear in the same order.

#     Parameters:
#     - df1: pandas DataFrame (first dataset)
#     - df2: pandas DataFrame (second dataset)

#     Returns:
#     - Aligned DataFrames (df1_aligned, df2_aligned)
#     """
#     # Ensure both DataFrames have the same index and are sorted by index
#     df1_aligned = df1.loc[df1.index.sort_values()]
#     df2_aligned = df2.loc[df1_aligned.index]  # Reorder df2 to match df1's index

#     # Check if indices are perfectly aligned (optional)
#     assert (df1_aligned.index == df2_aligned.index).all(), "Indices are not perfectly aligned!"

#     return df1_aligned, df2_aligned




def load_preprocess_data(data_path, cosmic_data_path, sep1, sep2, output_folder, output_filename, out_cosmic_filename):    
    
    if not os.path.exists(os.path.join(output_folder, output_filename)):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load data and COSMIC data 
        data = pd.read_csv(data_path, sep=sep1) 
        cosmic_data = pd.read_csv(cosmic_data_path, sep=sep2)

        # Set the index of the data to the first column (signature types)
        cosmic_data = cosmic_data.set_index(cosmic_data.columns[0])

        cosmic_aligned = cosmic_data.loc[cosmic_data.index.sort_values()]
        data_aligned = data.loc[cosmic_aligned.index]

        # Check if indices are perfectly aligned (optional)
        assert (cosmic_aligned.index == data_aligned.index).all(), "Indices are not perfectly aligned!"

        # Save the aligned data to a new file
        data_aligned.to_csv(os.path.join(output_folder, output_filename))
        print("Data saved to", os.path.join(output_folder, output_filename))

        if not os.path.exists(os.path.join(output_folder, out_cosmic_filename)):
            cosmic_aligned.to_csv(os.path.join(output_folder, out_cosmic_filename))
            print("COSMIC data saved to", os.path.join(output_folder, out_cosmic_filename))

        print("Data loaded and aligned successfully!")
        



    


