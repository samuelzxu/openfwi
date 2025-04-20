export MODEL_DIR="Invnet_models"
export MODEL_VARIATION_DIR="Invnet_models/invnet_all_baseline"

# Go to https://www.kaggle.com/settings, download your API token file and store it at ~/.kaggle/kaggle.json

# Create the model (which will hold the different variations).
kaggle models init -p $MODEL_DIR # This will create a skeleton model-metadata.json
vim $MODEL_DIR/model-metadata.json # Edit model metadata (name, slug, etc.)
kaggle models create -p $MODEL_DIR

# Create the model variation
# IMPORTANT, your model files (weights, config, etc.) should be inside the $MODEL_VARIATION_DIR folder.
kaggle models instances init -p $MODEL_VARIATION_DIR # This will create a skeleton model-instance-metadata.json
vim $MODEL_VARIATION_DIR/model-instance-metadata.json # Edit variation metadata (slug, framework, etc.)

kaggle models instances create -p $MODEL_VARIATION_DIR

# # To create a new version for an existing variation, use this command instead:
# kaggle models instances versions create -p $MODEL_VARIATION_DIR --version-notes "Made it better" $KAGGLE_USERNAME>/$MODEL_SLUG/$FRAMEWORK/$VARIATION_SLUG