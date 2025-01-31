library(stringr)
library(ggplot2)
library(tidyr)


current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))


data_path <- paste0( str_split( current_path, "artifact_code" )[[1]][1], 'artifact_code/data/'  )


T4 <- read.csv( paste0(data_path, 'disp_impact_table.csv') )

T4$Task <- factor(T4$Task,
                  levels=c("Anxiety", "Literacy", "Numeracy", "Trust", "Depression"))
T4$Demographic <- factor(T4$Demographic, 
                         levels=c('Age', 'Sex', 'Race', 'Educ.', 'Inc.', 'ESL'))


T4 %>%
  pivot_longer( c( starts_with("Rand"), starts_with("Hard") ),
                names_sep = "_", names_to = c("Sample","Bound") ) %>%
  pivot_wider( names_from = "Bound", values_from = "value" ) %>%
  ggplot() +
  geom_errorbar( aes(x=Demographic, color=Sample, linetype=Sample,
                     ymax=High, ymin=Low), position=position_dodge(width=0.5) ) + 
  coord_flip() + 
  geom_hline( aes(yintercept = 0.8), color="red", linetype="dotted" )+
  geom_hline( aes(yintercept = 1.2), color="red", linetype="dotted" )+
  facet_wrap(~Task) + 
  theme_bw() +
  theme(legend.position = "bottom") + 
  scale_color_brewer(palette="Dark2")

ggsave( "diPlot_v4.png", height=5, width=5, dpi = 180 )
