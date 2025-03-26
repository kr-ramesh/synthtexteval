PATH_TO_FOLDER=$1
# Create 
mkdir $PATH_TO_FOLDER/data

# For saving generator files and training data.
mkdir $PATH_TO_FOLDER/data/generator
mkdir $PATH_TO_FOLDER/data/generator/models
mkdir $PATH_TO_FOLDER/data/generator/data

# For storing synthetic outputs and real data for comparison
mkdir $PATH_TO_FOLDER/data/data

# For benchmarking results on downstream tasks
# Classification
mkdir $PATH_TO_FOLDER/data/benchmark
mkdir $PATH_TO_FOLDER/data/benchmark/classification
mkdir $PATH_TO_FOLDER/data/benchmark/classification/data
mkdir $PATH_TO_FOLDER/data/benchmark/classification/data/real_tab
mkdir $PATH_TO_FOLDER/data/benchmark/classification/data/test
mkdir $PATH_TO_FOLDER/data/benchmark/classification/models
mkdir $PATH_TO_FOLDER/data/benchmark/classification/test-results

# Coref/Mention Annotation
mkdir $PATH_TO_FOLDER/data/benchmark/coref
mkdir $PATH_TO_FOLDER/data/benchmark/coref/data
mkdir $PATH_TO_FOLDER/data/benchmark/coref/data/real_tab

# Privacy results
mkdir $PATH_TO_FOLDER/data/privacy


# For demo

mkdir $PATH_TO_FOLDER/data/benchmark/classification/data/test/tab
mkdir $PATH_TO_FOLDER/data/benchmark/classification/data/test/mimic
