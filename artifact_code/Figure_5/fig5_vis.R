library(tidyverse)
library(ggtext)
library(ggplot2)
library(GGally)
library(stringr)

current_path = rstudioapi::getActiveDocumentContext()$path 
base_path <- paste0( str_split( current_path, "artifact_code" )[[1]][1], 'artifact_code/'  )
setwd(base_path)


source('utils/plotting_helpers.R')


model_diff <- read.csv( 'data/model_diff.csv' )

comp_cols <- c( 'boundary_dist', 'losses', 'pvi', 'times_forgotten',
                'instance_hardness', 'irt_difficulty', 'tok_len' )

# request to see the entire correlation plot across complexity metrics
# WARNING: takes ~5min
ggpairs( model_diff %>% select(comp_cols) , progress=FALSE,
         upper = list(continuous = wrap("cor", method = "spearman") ),
         lower = list(continuous = wrap("points", alpha = 0.02)), 
         labeller = as_labeller(color_labeller) ) +
  theme_minimal() +
  theme(strip.text = element_markdown(),
        legend.position = 'bottom')

ggsave( "Figure_5/micro_pairplot_v2.png", height=12, width=16, dpi = 180 )
