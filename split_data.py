import os
import shutil
import splitfolders

input_dir = "input/wikiart"
output_dir = "input/processed_data"

# there is no "Renaissance" directory to merge all of types so its created
os.mkdir(input_dir + '/Renaissance')

# merge
dirs = os.listdir(input_dir)

for dir in [temp for temp in dirs if '_' in temp]:
    splitted = dir.split('_')

    for word in splitted:
        if word in dirs:
            for file in os.listdir(input_dir + '/' + dir):
                original = input_dir + '/' + dir + '/' + file
                target = input_dir + '/' + word + '/' + file
                shutil.move(original, target)

                print(dir + '/' + file + ' -> ' + word + '/' + file)

            os.rmdir(input_dir + '/' + dir)
            print('Removed ' + dir)
            break


# reduce
dirs = os.listdir(input_dir)

for dir in dirs:
    num_of_files = len(os.listdir(input_dir + '/' + dir))

    if num_of_files < 5000:
        shutil.rmtree(input_dir + '/' + dir)
        print('Removed ' + dir + ' (< 5000 images)')

# split
print('Splitting data...')
splitfolders.ratio(input_dir, output_dir, seed=1337, ratio=(0.6, 0.2, 0.2))
