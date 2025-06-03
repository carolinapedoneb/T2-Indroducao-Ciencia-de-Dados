##########Correlation
cor_matrix = cor(data[,c(2:6)])
library(corrplot)

corrplot(cor_matrix, method = "color", 
         col = colorRampPalette(c("lightblue", "white", "yellow"))(200),
         type = "full", order = "hclust",
         addCoef.col = "black", # Add correlation coefficients
         tl.col = "black", tl.srt = 45, # Text label color and rotation
         number.cex = 0.7, # Size of correlation coefficients
         title = "Heatmap of Correlation Matrix",
         mar = c(0,0,1,0)) # Adjust the margins
         

###All pairwise comparisons
pairs(data[,c(2:6)], pch=20)