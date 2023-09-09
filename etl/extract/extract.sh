#extract data from s3 bucket
zipfile="dga.zip"
curl https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/y8ph45msv8-1.zip -o "$zipfile"


#create path of where to place the data contained in a zip file
dest_dir="../../data/bronze"
mkdir -p $dest_dir

#move the data to the path
mv "$zipfile" "$dest_dir"

#unzip the contents
unzip -q "$dest_dir/$zipfile" -d "$dest_dir"

#remove uneeded directories
top_folder="UMUDGA - University of Murcia Domain Generation Algorithm Dataset"
rm -rf "$dest_dir/$top_folder/Domain Generation Algorithms"
rm -rf "$dest_dir/$top_folder/Language Data"
rm -rf "$dest_dir/$top_folder/Other packages"

#for the remaining directory,  loop through each folder

source_dir="$dest_dir/$top_folder/Fully Qualified Domain Names"

for folder in "$source_dir"/*; do

    # Check if the current item is a directory
    if [ -d "$folder" ]; then

        # Find the 1000000.txt file in the current folder
        list_file="$folder/list/1000000.txt"

        # Check if the txt file exists in the current folder
        if [ -f "$list_file" ]; then

            # Extract the directory name (folder name) and construct a unique filename
            folder_name=$(basename "$folder")
            unique_filename="$dest_dir/${folder_name}_list.txt"

            # Copy the txt file to the destination with a unique name
            cp "$list_file" "$unique_filename"
        fi
    fi
done

rm -rf "$dest_dir/$top_folder/"
rm "$dest_dir/$zipfile"

#extract top million websites
curl https://downloads.majestic.com/majestic_million.csv -o "$dest_dir/top_million_benign.csv"


