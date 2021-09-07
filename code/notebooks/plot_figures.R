suppressMessages(library(arrow))
suppressMessages(library(tidyverse))
suppressMessages(library(RColorBrewer))
suppressMessages(library(scales))
suppressMessages(library(viridis))
suppressMessages(library(lemon))

get_pair_ranks <- function(ds, f, c) {
    path <- paste('../../data/analysis/duplicate_pair_ranks/', ds, '/', f, '_', c, '.parquet', sep = '')
    read_parquet(file = path) %>% 
        mutate(dataset=ds, feat=f, col=c)
}

default_theme <- function() {
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    plot.title = element_text(size=20, family='', hjust=0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.text.x=element_text( size=20,family=""),
    axis.text.y=element_text( size=20,family=""),
    axis.title.x=element_text(size=20,family=""),
    axis.title.y=element_text(size=20,family=""),
    axis.ticks.length=unit(.25, "cm"),
    legend.text=element_text( size=20, family=""),
    legend.key = element_rect(
      fill = "white",
      color="white",
      size=2),
    legend.position = "bottom",
    strip.text = element_text(size =20),
    strip.background = element_rect(fill="#FFFFFF"))
}

save <- function(plot, filename, w=2250, h=1500) {
  ggsave(filename, plot=plot, width=w/300, height=h/300, dpi="print")
}

similarity_boxplots <- function(save_fig=FALSE) {
    so_samples <- c(0:4) %>% map(function(i) paste('so_samples/sample_', i, sep=''))
    datasets <- c('gamedev_se', 'gamedev_so', so_samples)
    feats <- c('jaccard','tfidf','bm25','doc2vec','topic','bertoverflow','mpnet')
    cols <- c('title', 'body', 'tags', 'title_body', 'title_body_tags', 'title_body_tags_answer')
        
    df <- datasets %>% 
    map(function(ds)
        feats %>% 
        map(function(f)
            cols %>% 
            map(function(c) get_pair_ranks(ds, f, c))) %>% 
        bind_rows) %>% 
        bind_rows %>% 
        mutate(
            dataset=ifelse(grepl('so_sample', dataset), 'so_sample', dataset),
            col=fct_rev(factor(col, cols)),
            feat=factor(feat, feats),
            col=fct_recode(col,
                           '1'='title',
                           '2'='body',
                           '3'='tags',
                           '4'='title_body',
                           '5'='title_body_tags',
                           '6'='title_body_tags_answer'),
            feat=fct_recode(feat,
                            'Jaccard'='jaccard',
                            'BM25'='bm25',
                            'TF-IDF'='tfidf',
                            'Doc2Vec'='doc2vec',
                            'Topic'='topic',
                            'BERTOverflow'='bertoverflow',
                            'MPNet'='mpnet'
                           ),
            dataset=fct_recode(dataset, 'Game Dev. SE'='gamedev_se', 'Game Dev. SO'='gamedev_so', 'General Dev. SO'='so_sample')
        ) %>% 
        group_by(dataset, feat, col) %>% 
        mutate(log10_rank = log10(rank)) %>% 
        mutate(outlier = log10_rank < quantile(log10_rank, .25) - 1.50*IQR(log10_rank)) %>% 
        ungroup
            
    plot <- df %>% 
        ggplot(aes(x=rank, y=col, fill=col, color=col)) +
        geom_boxplot(aes(group=col), outlier.shape = NA) +
        geom_point(data = function(x) filter(x, outlier), position = position_jitter(w = 0, h = 0.05), shape=".", alpha=0.8) +
        facet_rep_grid(rows=vars(feat), cols=vars(dataset), scales='free_x') +
        annotation_logticks(sides='b', outside=TRUE) +
        coord_cartesian(clip = "off") +
        scale_x_log10(limits = c(1, 300000), breaks=c(10, 1000, 100000), labels=c('10', '1000', '100K')) +
        default_theme() + 
        scale_fill_viridis_d(alpha=0.5) +
        scale_color_viridis_d() +
        theme(axis.line=element_line()) +
        labs(
          x="",
          y="") +
        theme(legend.position="none")
                   
    if(save_fig) {
        save(plot, '../../figures/duplicate_pair_ranks_boxplots.pdf', w=4000, h=5500)
    }
                   
    plot
}
