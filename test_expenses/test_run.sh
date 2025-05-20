#!/bin/bash

# Run the expense automation script with our improvements
cd ..
echo "Testing expense automation with enhancements"
python expenses_automation.py test_expenses

# Compare with how many files were processed
echo "Processing complete!" 