print("Starting R ...")
#library("rlang")
#print(packageVersion("rlang"))

library(withr , lib.loc= "/doctorai/niccoloc/libR2")
library(dplyr, lib.loc= "/doctorai/niccoloc/libR2")
library(tidyr, lib.loc= "/doctorai/niccoloc/libR2")
library(viridisLite, lib="/doctorai/niccoloc/libR2")
library(farver, lib="/doctorai/niccoloc/libR2")
library(labeling, lib="/doctorai/niccoloc/libR2")
# library(foreach, lib.loc= "/doctorai/niccoloc/libR2")
library(ggplot2, lib.loc= "/doctorai/niccoloc/libR2")
library(stringr, lib.loc= "/doctorai/niccoloc/libR2")
# library(patchwork , lib.loc = "/doctorai/niccoloc/libR2")
library(data.table , lib.loc = "/doctorai/niccoloc/libR2")
library(ggrepel , lib.loc= "/doctorai/niccoloc/libR2")
print("Library loaded correctly !")

# Retrieving command line arguments
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 3) {
  stop("Not enough arguments! Expecting paths for ED_input, LD_input, and output_path.")
}

# Assign arguments to variables
ED_input <- args[1]
LD_input <- args[2]
output_path <- args[3]

# 
# ED_input='/doctorai/niccoloc/airr_atlas_bk/Vicinity_results/Tz_cosine/summary_results_ED_Tz_cosine.csv'
# LD_input ='/doctorai/niccoloc/airr_atlas_bk/Vicinity_results/Tz/d_mean1_summary_LD_Tz_10k.csv'


#tmp_ed_sum<- fread("summary_results_ED_AG_SPEC.csv")[,-1] %>% as.data.frame()
#tmp_ed_sum<- fread("summary_results_ED_Ag_spec_whole_PRELIMINARY_AB2.csv")[,-1] %>% as.data.frame()
#d_mean1<- fread("d_mean1_summary_results_LD_AG_SPEC.csv") %>% as.data.frame()
tmp_ed_sum<- fread(ED_input)[,-1] %>% as.data.frame()
d_mean1<- fread(LD_input) %>% as.data.frame()
labels=d_mean1[,'affinity']
threholds= tmp_ed_sum$Threshold
# get the max_LD used in the input
max_LD=length(grep("Perc_LD",colnames(d_mean1)))
# preprocess the data
LD_perc <- d_mean1[paste0("Perc_LD_", 1:max_LD)]     %>% as.data.frame() %>% dplyr::mutate(affinity=d_mean1[,'affinity'])                
LD_points <- d_mean1[paste0("Avg_Num_LD_", 1:max_LD)]  %>% as.data.frame() %>% dplyr::mutate(affinity=d_mean1[,'affinity'])           
LD_NULL_perc <- (d_mean1[paste0("Avg_NaN_Percentage_LD_", 1:max_LD)]*100) %>% as.data.frame() %>% dplyr::mutate(affinity=d_mean1[,'affinity'])     

ED_perc <- tmp_ed_sum[paste0("Perc_", labels)]     %>% as.data.frame()     %>% mutate(thr=threholds)       
ED_points <- tmp_ed_sum[paste0("AvgPoints_", labels)]  %>% as.data.frame()   %>% mutate(thr=threholds)  
ED_NULL_perc <- tmp_ed_sum[paste0("NULLPerc_", labels)] %>% as.data.frame()    %>% mutate(thr=threholds)  
ED_LD_sim <- tmp_ed_sum[paste0("LD_avgSim_", labels)] %>% as.data.frame()    %>% mutate(thr2=threholds)  


df_LD=bind_cols(
  tidyr::pivot_longer(LD_perc,starts_with("Perc"), values_to = "Vicinity"),
  tidyr::pivot_longer(LD_points,starts_with("Avg"), values_to = "AvgNN")[,-1]
  ) 
df_ED=bind_cols(
  tidyr::pivot_longer(ED_perc,starts_with("Perc"), values_to = "Vicinity", names_to="affinity"),
  tidyr::pivot_longer(ED_points,starts_with("Avg"), values_to = "AvgNN" )[,-1],
  tidyr::pivot_longer(ED_LD_sim,starts_with("LD_avgSim_"), values_to = "LD_sim"),
)  %>% mutate(affinity= str_remove(affinity, "Perc_"))


p_avg<-ggplot()+
  geom_point(data=df_LD, aes(AvgNN,Vicinity, ), color= "red", shape=1, size=1)+
  geom_point(data=df_ED, aes(AvgNN,Vicinity,color= LD_sim ),  size=3)+
  geom_text_repel(data=df_ED, aes(AvgNN,Vicinity, label= paste0(thr,"_",round(LD_sim,1) )),min.segment.length = 0, max.overlaps = 5)+
  geom_line(data=df_LD, aes(AvgNN,Vicinity, ),color= "red",  shape=4 , size=1, alpha=0.5)+ 
  geom_line(data=df_ED, aes(AvgNN,Vicinity, ),color= "blue",  shape=4 , size=1, alpha=0.5)+
  scale_color_viridis_c( option = "F" , direction =-1)+
  facet_wrap(vars(affinity), scales = "free_x")+
  theme_bw()


# --------------------------PERC NULL --------------------------------


df_LD_NULL=bind_cols(
  tidyr::pivot_longer(LD_perc,starts_with("Perc"), values_to = "Vicinity"),
  tidyr::pivot_longer(LD_NULL_perc,starts_with("Avg_NaN"), values_to = "NULL_p")[,-1]
) 
df_ED_NULL=bind_cols(
  tidyr::pivot_longer(ED_perc,starts_with("Perc"), values_to = "Vicinity", names_to="affinity"),
  tidyr::pivot_longer(ED_NULL_perc,starts_with("NULL"), values_to = "NULL_p" )[,-1]
)  %>% mutate(affinity= str_remove(affinity, "Perc_"))


p_null<-ggplot()+
  geom_point(data=df_LD_NULL, aes(NULL_p,Vicinity, ), color= "red", shape=1, size=1)+
  geom_point(data=df_ED_NULL, aes(NULL_p,Vicinity, ),color= "blue",  shape=1, size=1)+
  geom_text_repel(data=df_ED_NULL, aes(NULL_p,Vicinity, label= thr ),min.segment.length = 0, max.overlaps = 5, force_pull = 10)+
  geom_line(data=df_LD_NULL, aes(NULL_p,Vicinity, ),color= "red",  shape=4 , linewidth=1, alpha=0.5)+ 
  geom_line(data=df_ED_NULL, aes(NULL_p,Vicinity, ),color= "blue",  shape=4 , linewidth=1, alpha=0.5)+
  facet_wrap(vars(affinity), scales = "free_x")+
  theme_bw()

# bind_rows(
#   df_ED %>% left_join(df_ED_NULL ,by=c('thr','affinity','Vicinity')) %>% mutate(metric ='ED'),
#   df_LD %>% left_join(df_LD_NULL ,by=c('affinity','name...2','Vicinity')) %>% mutate(metric ='LD') ) %>% 
#   filter(!is.na(Vicinity)) %>% 
#   
#   mutate(Loneliness = (1- (NULL_p/100))) %>% 
#   mutate(Vicinity_corr = Vicinity * Loneliness) %>% View()
#   ggplot(.)+
#   geom_point( aes(AvgNN,Vicinity_corr, color = metric),  shape=1, size=1)+
#   geom_line( aes(AvgNN,Vicinity_corr, color = metric),   shape=4 , linewidth=1, alpha=0.5)+ 
#   facet_wrap(vars(affinity), scales = "free_x")+
#   theme_bw()

#output_path="./Ag_whole"
print(output_path)
dir.create(output_path)
title1=str_c(output_path,"/vicinity_avgNN.jpeg")
title2=str_c(output_path,"/vicinity_NULL.jpeg")

fwrite(df_LD,str_c(output_path,"/LD_plot_input.tsv"))
print(title1)
title2

ggsave( p_avg  , filename = title1 ,  device = "jpeg",
         width = 45,height =30, units = "cm")  
ggsave( p_null , filename = title2 ,  device = "jpeg",
         width = 45,height =30, units = "cm")


