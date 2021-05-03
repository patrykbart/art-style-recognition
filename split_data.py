import splitfolders

input_folder = "input/wikiart"
output_folder = "input/processed_data"
splitfolders.ratio(input_folder, output_folder, seed=1337, ratio=(0.6, 0.2, 0.2))
