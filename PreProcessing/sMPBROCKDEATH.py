import pandas as pd 

mpb = pd.read_csv("/Users/carolbarroco/Documents/RecomendSisKNN/sMPB.csv")
rock1 = pd.read_csv("/Users/carolbarroco/Documents/RecomendSisKNN/sROCK1.csv")
rock2 = pd.read_csv("/Users/carolbarroco/Documents/RecomendSisKNN/sROCK2.csv")
heavy_metal = pd.read_csv("/Users/carolbarroco/Documents/RecomendSisKNN/sHEAVYMETAL.csv")

mpb["track_genre"] = "mpb"
rock1["track_genre"] = "rock"
rock2["track_genre"] = "rock"
heavy_metal["track_genre"] = "heavy_metal"
print(rock1)


mpbrockdeathm = pd.concat([mpb,rock1,rock2,heavy_metal],ignore_index=True)
mpbrockdeathm = mpbrockdeathm.drop_duplicates()
print(len(mpbrockdeathm))
mpbrockdeathm.to_csv("sMPBROCKMETAL_validation.csv")