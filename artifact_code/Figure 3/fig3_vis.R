library(ggplot2)
library(dplyr)
library(tidyverse)
library(ggtext)
library(gridExtra)


# local path management
current_path = rstudioapi::getActiveDocumentContext()$path 
base_path <- paste0( str_split( current_path, "artifact_code" )[[1]][1], 'artifact_code/'  )
setwd(base_path)



source('utils/plotting_helpers.R')


# NOTE; this includes WER
# NOTE; NAs come from NA scorrs -- which correctly return NA for metrics with zero variance
# data can be produced from fig3_preprocess.ipynb
micro_corr_df_wer <- read.csv( 'data/micro_corr_df_wer.csv' )


comp_lst <- c( 'BP', 'Loss', 'PVI', 'TF', 'PH', 'IRT', 'SL' )
names(comp_lst) <- c( 'boundary_prox', 'losses', 'pvi', 'times_forgotten',
                      'instance_hardness', 'irt_difficulty', 'tok_len' )


this_plot <- produce_scorr_fig( micro_corr_df_wer, sort='paper' )


ggsave( "Figure 3/micro_scorr_inc_wer_v2.png", height=5, width=5, dpi = 180 )


