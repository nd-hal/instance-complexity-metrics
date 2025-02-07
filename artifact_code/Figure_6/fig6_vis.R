library(ggplot2)
library(dplyr)
library(tidyverse)
library(ggtext)
library(gridExtra)
library(ggpubr)


# local path management
current_path = rstudioapi::getActiveDocumentContext()$path 
base_path <- paste0( str_split( current_path, "artifact_code" )[[1]][1], 'artifact_code/'  )
setwd(base_path)

source('utils/plotting_helpers.R')



# NOTE; Depression data with 'wer' (Word Error Rate) score variable not publicly available via IRB
## data can be produced from fig3_preprocess.ipynb
# NOTE; NAs come from NA scorrs -- which correctly return NA for metrics with zero variance
micro_corr_df_wer <- read.csv( 'data/micro_corr_df_wer.csv' )


comp_lst <- c( 'BD', 'Loss', 'PVI', 'TF', 'PH', 'IRT', 'SL' )
names(comp_lst) <- c( 'boundary_dist', 'losses', 'pvi', 'times_forgotten',
                      'instance_hardness', 'irt_difficulty', 'tok_len' )

# these_svars <- c( 'Anxiety', 'Numeracy', 'Literacy', 'Trust', 'Depression' )
# names(these_svars) <- c( 'Anxiety', 'Numeracy', 'SubjectiveLit', 'TrustPhys', 'wer'  )
these_svars <- c( 'Anxiety', 'Numeracy', 'Literacy', 'Trust' )
names(these_svars) <- c( 'Anxiety', 'Numeracy', 'SubjectiveLit', 'TrustPhys'  )


these_plots <- list()
for( sname in names(these_svars) ){
  this_svar <- these_svars[sname]
  this_filepath <- paste0( 'data/corr_by_svar/micro_corr_df-for_', sname, '.csv' )
  print( this_filepath  )
  this_corr_df <- read.csv( this_filepath )
  
  this_p <- produce_scorr_fig( this_corr_df, sort='paper',
                               this_svar=this_svar )
  assign( paste0(this_svar, '_plot'), this_p )
  
  these_plots <- c( these_plots, this_p )
  
}



# for better formatting when faceted
Anxiety_plot <- Anxiety_plot +
  scale_color_discrete(guide='none')+
  theme(axis.title.x=element_blank())
Numeracy_plot <- Numeracy_plot +
  scale_color_discrete(guide='none')+
  theme(axis.title.x=element_blank())
Literacy_plot <- Literacy_plot +
  scale_color_discrete(guide='none')+
  theme(axis.title.x=element_blank())
# Depression_plot <- Depression_plot +
#   scale_color_discrete(guide='none')+
#   theme(axis.title.x=element_blank())


# NOTE; missing WER plot due to IRB restrictions on data
# this will be a 2x2 instead of the 2x3 in the paper
# sv_fig <- ggarrange(Anxiety_plot, Numeracy_plot, Literacy_plot,
#                     Trust_plot, Depression_plot,
#                     ncol = 3, nrow = 2)

sv_fig <- ggarrange(Anxiety_plot, Numeracy_plot, Literacy_plot, Trust_plot,
                    ncol = 2, nrow = 2)

ggsave( "Figure_6/micro_scorr_by_svar_v4.png", height=16, width=16, dpi = 180 )






