from sklearn.metrics import cohen_kappa_score

rater1_aas_scores = [4,4,5,5,3,4,2,3]
rater2_aas_scores = [4,4,5,3,4,3,3,2]

rater1_crs_scores = [4,4,5,4,3,2,3,2]
rater2_crs_scores = [3,4,4,3,3,1,2,1]
# Assuming you have the lists populated
kappa_aas = cohen_kappa_score(rater1_aas_scores, rater2_aas_scores)
kappa_crs = cohen_kappa_score(rater1_crs_scores, rater2_crs_scores)

print(f"Cohen's Kappa for AAS: {kappa_aas:.3f}")
print(f"Cohen's Kappa for CRS: {kappa_crs:.3f}")