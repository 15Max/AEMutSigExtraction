import os 
import pandas as pd


def load_preprocess_data(data_path, cosmic_data_path, sep1, sep2, output_folder, output_filename):    
    
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

        print("Data loaded and aligned successfully!")
    
    else:
        print("Data already exists in ", os.path.join(output_folder, output_filename))



    


