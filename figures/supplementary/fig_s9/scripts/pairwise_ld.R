library(tidyr)
library(dplyr)
library(ggplot2)
library(ggview)
library(readr)
library(ggridges)
library(scales)
df = read_csv("../data/tz_pairwise_ld_frequencies.csv") %>%
    mutate(
        cls = case_when(
            cls == "hb/hb" ~ "HB-HB",
            cls == "hb/lb" ~ "HB-LB",
            cls == "lb/lb" ~ "LB-LB"))

df$cls = factor(df$cls, levels=c("HB-HB", "LB-LB", "HB-LB"))
colors = c("HB-HB" = "#8ECAE6", "LB-LB" = "#023047", "HB-LB" = "#FB8500")

bar_plot = df %>%
    ggplot(aes(x = distance, y = count, fill = cls, color = cls)) +
    geom_bar(stat="identity", position = position_dodge(width = 0.90), width = 0.75,alpha =0.7) +
    scale_x_continuous(breaks = breaks_pretty(n = 10)) +
    scale_color_manual(values = colors) +
    scale_fill_manual(values = colors) +
    theme_minimal() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom"
    ) +
    labs(x = "Levenshtein distance", y = "Number of sequences", fill = "Binding labels of variant pair", color = "Binding labels of variant pair") +
    canvas(width = 6, height = 4, dpi = 300)
bar_plot
bar_plot %>% save_ggplot("../output/fig_s9_pairwise_ld.png")
