library(stringr)
library(glue)

create_color_labs <- function(these_metrics){
  # map metrics to colors
  comp_cols = c( 'boundary_prox', 'losses', 'pvi', 'times_forgotten',
                 'instance_hardness', 'irt_difficulty', 'tok_len' )
  comp_cols = c( 'BP', 'Loss', 'PVI', 'TF', 'PH', 'IRT', 'SL' )
  met_to_col_lst = list( 'darkred', 'navy', 'darkred', 'navy',
                         'navy', 'navy', 'navy' )
  names(met_to_col_lst) <- comp_cols
  
  colored_metrics <- c()
  for(s in these_metrics){
    # check if multiple metrics (i.e., correlation)
    if(grepl( ':', s, fixed = TRUE)){
      split_met <- str_split(s, ' : ')[[1]]
      
      met1 <- split_met[1]
      met1_col <- met_to_col_lst[[ met1 ]]
      met2 <- split_met[2]
      met2_col <- met_to_col_lst[[ met2 ]]
      
      col_s <- glue( "<span style='color:{met1_col}'>{met1}</span> : <span style='color:{met2_col}'>{met2}</span>" )
    } else {
      met <- s
      met_col <- met_to_col_lst[[ met ]]
      col_s <- glue( "<span style='color:{met_col}'>{met}</span>" )
    }
    
    colored_metrics <- c( colored_metrics, col_s )
    
  }
  return(colored_metrics) 
}



color_labeller <- function(mets){
  comp_cols = c( 'boundary_prox', 'losses', 'pvi', 'times_forgotten',
                 'instance_hardness', 'irt_difficulty', 'tok_len' )
  comp_cols = c( 'BP', 'Loss', 'PVI', 'TF', 'PH', 'IRT', 'SL' )
  
  col_mets <- c()
  for(met in mets){
    if(met %in% comp_cols){
      # map metrics to colors
      met_to_col_lst = list( 'darkred', 'navy', 'darkred', 'navy',
                             'navy', 'navy', 'navy' )
      names(met_to_col_lst) <- comp_cols
      
      met_col <- met_to_col_lst[[ met ]]
      col_s <- glue( "<span style='color:{met_col}'>{met}</span>" )
      col_mets <- c( col_mets, col_s  )
      
    } else {
      col_mets <- c( col_mets, met  )
    }
  }
  
  return(col_mets)
}




get_lower_tri<-function(cormat){
  cormat[lower.tri(cormat)] <- NA
  return(cormat)
}


plot_cor_mat <- function(in_df, corr_method='pearson',
                         use_small_text=FALSE){
  library(reshape2)
  library(ggplot2)
  cormat <- cor(in_df, method=corr_method, use="complete.obs")

  lower_tri <- get_lower_tri(cormat)
  melted_cormat <- melt(lower_tri, na.rm = TRUE) %>% mutate(value=round(value, 2))
  
  text_size <- if (use_small_text) 3 else 6
  # Heatmap
  ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(low = "darkblue", high = "darkred", mid = "white",
                         midpoint = 0, limit = c(-1,1), space = "Lab",
                         name="Pearson\nCorrelation",
                         guide='none') +
    geom_text(aes(label=value), color='white',
              size=text_size,
              data=melted_cormat[melted_cormat$value < 1,]) +
    theme_minimal()+
    theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                     size = 3*text_size, hjust = 1),
          axis.text.y = element_text(angle = 0, vjust = 1,
                                     size = 3*text_size, hjust = 1))+
    coord_fixed() +
    theme(axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),
    )

}



plot_cross_cor_mat <- function(in_df, comp_cols, perform_cols,
                               corr_method='pearson',
                               use_small_text=FALSE){
  library(reshape2)
  library(ggplot2)
  cormat <- cor(in_df, method=corr_method, use="complete.obs")
  
  lower_tri <- get_lower_tri(cormat)
  melted_cormat <- melt(lower_tri, na.rm = TRUE) %>% mutate(value=round(value, 2))
  
  # fil_melted_cormat <- melted_cormat %>%
  #   filter( ( (Var1 != Var2) &
  #               !(Var1 %in% comp_cols & Var2 %in% comp_cols) ) )
  
  fil_melted_cormat <- melted_cormat %>% filter( (
    (Var1 %in% comp_cols & !(Var2 %in% comp_cols)) |
      ( !(Var1 %in% comp_cols) & Var2 %in% comp_cols)) )
  
  text_size <- if (use_small_text) 3 else 4
  # Heatmap
  ggplot(data = fil_melted_cormat, aes(Var2, Var1, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(low = "darkblue", high = "darkred", mid = "white",
                         midpoint = 0, limit = c(-1,1), space = "Lab",
                         name="Pearson\nCorrelation",
                         guide='none') +
    geom_text(aes(label=value), color='white',
              size=text_size,
              data=fil_melted_cormat[fil_melted_cormat$value < 1,]) +
    facet_grid(strat ~ score_var) +
    theme_minimal()+
    theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                     size = 3*text_size, hjust = 1),
          axis.text.y = element_text(angle = 0, vjust = 1,
                                     size = 3*text_size, hjust = 1))+
    coord_fixed() +
    theme(axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),
    )
  
}



plot_facet_cross_cor_mat <- function(in_df, comp_cols, perform_cols,
                                     corr_method='pearson',
                                     use_small_text=FALSE){
  library(reshape2)
  library(ggplot2)
  
  all_melted_cormats <- data.frame(matrix(ncol =5, nrow = 0))
  colnames(all_melted_cormats) <- c( 'Var1', 'Var2', 'value', 'score_var', 'strat' )
  
  for(sv in unique(in_df$score_var)){
    for(st in unique(in_df$strat)){
      tdf <- in_df %>% filter(score_var==sv & strat==st)
      
      cormat <- cor(tdf[, c(met_cols, comp_cols)], method=corr_method, use="complete.obs")
      
      lower_tri <- get_lower_tri(cormat)
      melted_cormat <- melt(lower_tri, na.rm = TRUE) %>% mutate(value=round(value, 2))
      
      fil_melted_cormat <- melted_cormat %>% filter( (
        (Var1 %in% comp_cols & !(Var2 %in% comp_cols)) |
          ( !(Var1 %in% comp_cols) & Var2 %in% comp_cols)) )
      
      fil_melted_cormat$score_var <- sv
      fil_melted_cormat$strat <- st
      
      all_melted_cormats <- rbind(all_melted_cormats, fil_melted_cormat)
    }
  }
  
  text_size <- if (use_small_text) 3 else 4
  # Heatmap
  ggplot(data = all_melted_cormats, aes(Var2, Var1, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(low = "darkblue", high = "darkred", mid = "white",
                         midpoint = 0, limit = c(-1,1), space = "Lab",
                         name="Pearson\nCorrelation",
                         guide='none') +
    geom_text(aes(label=value), color='white',
              size=text_size,
              data=all_melted_cormats[fil_melted_cormat$value < 1,]) +
    facet_grid(strat ~ score_var) +
    theme_minimal()+
    theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                     size = 3*text_size, hjust = 1),
          axis.text.y = element_text(angle = 0, vjust = 1,
                                     size = 3*text_size, hjust = 1))+
    coord_fixed() +
    theme(axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),
    )
  
  
  
}






library(tidyr)
library(ggplot2)


format_print_metrics <- function(in_corr_df){
  micro_corr_df_print <- in_corr_df %>%
    separate(metrics, c("met1", "met2"), " : ") %>%
    mutate(met1=comp_lst[met1], met2=comp_lst[met2]) %>%
    mutate(metrics=paste0(met1, ' : ', met2) )
  
  return(micro_corr_df_print)
}


produce_scorr_fig <- function(in_corr_df, sort='alpha', this_col='black', this_svar=NULL) {
  # map metric names
  comp_lst <- c( 'BP', 'Loss', 'PVI', 'TF', 'PH', 'IRT', 'SL' )
  names(comp_lst) <- c( 'boundary_prox', 'losses', 'pvi', 'times_forgotten',
                        'instance_hardness', 'irt_difficulty', 'tok_len' )
  # format to print
  micro_corr_df_print <- format_print_metrics( in_corr_df )
  
  # sort metrics to ensure properly ordered labels
  # sorted_micro_met <- (micro_corr_df_print %>% arrange( mean_scorr ))$metrics
  if(sort=='alpha'){
    sorted_micro_met <- (micro_corr_df_print %>% arrange( metrics ))$metrics
  } else if(sort=='paper') {
    paper_corr_df_print <- format_print_metrics( micro_corr_df_wer )
    sorted_micro_met <- (paper_corr_df_print %>% arrange( mean_scorr ))$metrics
  } else {
    print( 'invalid sort arg not in {"alpha", "paper"}'  )
    return(NULL)
  }
  
  
  p <- micro_corr_df_print %>%
    mutate(metrics=ordered(metrics, levels=sorted_micro_met) ) %>%
    rename(`Micro-averaged Spearman Correlation` = mean_scorr) %>%
    ggplot() +
    geom_point(aes(x=`Micro-averaged Spearman Correlation`, y=metrics), color=this_col) +
    geom_errorbarh(aes(y=metrics, xmin=ci_low, xmax=ci_high), color=this_col) +
    geom_vline(xintercept=0, color='dark red', alpha=0.7) +
    # NOTE; pink line is a "weak" Spearman correlation
    geom_vline(xintercept=0.2, color='pink', alpha=0.6) +
    geom_vline(xintercept=-0.2, color='pink', alpha=0.6) +
    # NOTE; orange line is a "moderate" Spearman Correlation
    geom_vline(xintercept=0.4, color='orange', alpha=0.3) +
    geom_vline(xintercept=-0.4, color='orange', alpha=0.3) +
    xlim(-.6, .6) +
    scale_color_manual(values = c("navy", "darkred"),
                       labels = c("Proportional to Complexity",
                                  "Inv. Prop. to Comp."),
                       name='color') +
    # arbitrary to make legend show up
    geom_point(aes(x=rep(0, nrow(micro_corr_df_print)), y=rep(0, nrow(micro_corr_df_print)), color = "Proportional to Complexity")) +
    geom_point(aes(x=rep(0, nrow(micro_corr_df_print)), y=rep(0, nrow(micro_corr_df_print)), color = "Inv. Prop. to Complexity")) +
    
    theme_minimal() +
    # ggtitle('Micro-Averaged Spearman Correlation') +
    scale_y_discrete(labels = create_color_labs(sorted_micro_met)) +
    theme(axis.text.y = element_markdown(size=12),
          axis.title.y = element_blank(),
          axis.title.x = element_text(vjust=-1),
          legend.position = 'bottom')
  
  if(!is.null(this_svar)){ 
    p <- p + ggtitle(this_svar)  
  }
  
  return(p)
}
