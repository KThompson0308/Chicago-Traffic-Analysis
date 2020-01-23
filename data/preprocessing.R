################
# File for Preprocessing data in R
# Author: Kevin Thompson
# Last Updated: January 22, 2019
################

library(tidyverse)
library(reticulate)
source_python("preprocessing.py")
crashes <- read_crashes("TrafficCrashesChicago.csv")
