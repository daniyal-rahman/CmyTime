#!/bin/bash

# This script generates the NeuroML and LEMS files for the worm model
# using the parameters specified in the project's config file.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
CONFIG_FILE="worm_rl_project/config/default.yaml"
C302_PATH="c302"
OUTPUT_DIR="examples" # c302 generates into this directory by default

# --- Helper function to parse YAML ---
# A simple parser using grep and sed. Assumes "key: value" format.
parse_yaml() {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
         -e "s|^\($s\)\(w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
         -e "s|^\($s\)\(w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

# --- Main script ---

echo "Parsing configuration from $CONFIG_FILE..."

# Read parameters from the YAML config file
eval $(parse_yaml $CONFIG_FILE "config_")

# Extract the specific values we need
PARAM_SET=${config_neural_sim_settings_parameter_set}
DURATION_S=${config_neural_sim_settings_duration}
DT_S=${config_neural_sim_settings_dt}

# Convert to milliseconds for c302
DURATION_MS=$(echo "$DURATION_S * 1000" | bc)
DT_MS=$(echo "$DT_S * 1000" | bc)

echo "Configuration loaded:"
echo "  Parameter Set: $PARAM_SET"
echo "  Duration: ${DURATION_MS}ms"
echo "  Timestep (dt): ${DT_MS}ms"

# Ensure the c302 library is in the Python path
export PYTHONPATH=${PYTHONPATH}:$(pwd)

# Generate the LEMS simulation files for the specified muscle network
# We target the 'Muscles' configuration of c302
echo "Generating NeuroML/LEMS model for c302..."
python -m c302.c302_Muscles "$PARAM_SET" -duration "$DURATION_MS" -dt "$DT_MS"

# The output files will be in the 'examples' directory by default.
# We can add move/rename commands here if needed, but for now,
# the config file points to the expected output location.
echo "Model generation complete. Files are located in the '$OUTPUT_DIR' directory."
echo "Main LEMS file: $OUTPUT_DIR/LEMS_c302_${PARAM_SET}_Muscles.xml"

exit 0
