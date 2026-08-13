import pandas as pd
#
# python phase1_check.py
#
#查明究竟是哪些被误判
p = pd.read_csv( 
r"D:\\00-FLagship Project\\results\\phase1_Conv1D_3layer(3dropout)_TCN_8Resblk_waist6\\loso_predictions.csv"
)

pd.crosstab(
    [p["true_label"], p["scenario"]],
    p["predicted_label"],
    margins=True
).to_csv(r"D:\\00-FLagship Project\\results\\phase1_Conv1D_3layer(3dropout)_TCN_8Resblk_waist6\\how_they_misclassified.csv", 
         index=True)

print(pd.crosstab(
    [p["true_label"], p["scenario"]],
    p["predicted_label"],
    margins=True
))


