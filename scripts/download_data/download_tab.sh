# Enter path to store TAB data
PATH_TO_FOLDER=$1
PATH_TO_FOLDER=${PATH_TO_FOLDER:-"../../data/"}
echo "Downloading TAB dataset and formatting it..."
python download_tab.py "${PATH_TO_FOLDER}generator/data/tab/" "${PATH_TO_FOLDER}/data/"