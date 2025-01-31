library(ggplot2)
library(dplyr)
library(tidyverse)


# local path management
current_path = rstudioapi::getActiveDocumentContext()$path 
base_path <- paste0( str_split( current_path, "artifact_code" )[[1]][1], 'artifact_code/'  )
setwd(base_path)

source('utils/plotting_helpers.R')



test_perform_full <- read.csv( 'data/test_perform_full.csv' )


ord_mets <- c( 'test_auc', 'test_acc', 'test_f1', 'irt_ability',
               'Sex_NonMale.DI', 'Age_Senior.DI', 'Race_POC.DI', 'Income_Low.DI',
               'Education_Low.DI', 'ESL.DI')
ord_names <- c( 'AUC', 'Acc.',  'F1 Score', 'IRT Ability',
                'Sex ADI', 'Age ADI', 'Race ADI', 'Educ. ADI', 'Inc. ADI', 'ESL ADI')


ord_mets[1:4]

perform_met_names <- ord_names
names(perform_met_names) <- ord_mets

strat_names <- c( 'Boundary-hard', 'Random' )
names(strat_names) <- c( 'Constant', 'None' )


strat_names <- c( 'Anxiety', 'Numeracy', 'Literacy', 'Trust', 'Depression' )
names(strat_names) <- c( 'Anxiety', 'Numeracy', 'SubjectiveLit', 'TrustPhys', 'wer' )



gath_perform <- test_perform_full %>%
  mutate(irt_ability =  ( (irt_ability - min(irt_ability)) / (max(irt_ability) - min(irt_ability)) )) %>%
  gather( metric, value,  -c(score_var, strat, model_name, model_type) )


gath_perform %>%
  filter(metric %in% ord_mets[1:4]) %>%
  mutate(metric = perform_met_names[metric],
         strat = strat_names[strat]) %>%
  rename(Sampling = strat) %>%
  ggplot() +
  geom_point(aes(x=score_var, y=value, color=`Sampling`, fill=`Sampling`),
             alpha=0.6) +
  facet_grid( ~metric ) +
  scale_color_manual(values=c('darkgreen', 'orange'))+
  scale_fill_manual(values=c('darkgreen', 'orange'))+
  theme_minimal() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 60, vjust = 1, hjust=1),
        axis.title.y = element_blank(),
        legend.position = 'bottom')


ggsave( "Appendix A9/perform_by_met_v2.png", height=5, width=8, dpi = 180 )

