import pandas as pd

path_annotation = "./data/annotation_global.csv"
path_meth = "./data/Matrix_meth.csv"


def union(row):
    return str(row["is_tumor"]) + "-" + row["project_id"]

annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)
#data = pd.read_csv(path_annotation, sep='\t')
data=y
print("annotation")
print("shape :" + str(data.shape))
print("with tumor: " + str(sum(data["is_tumor"] == "tumor")))
print("# tumor types: " + str(len(data["project_id"].unique())))
data["newlabel"] = data.apply(lambda row: union(row), axis=1)
for type in data["project_id"].unique():
    print("# of " + type + ": " + str(sum(data["project_id"] == type)))
print("--------------------------------------------------------")
for type in data["newlabel"].unique():
    print("# of " + type + ": " + str(sum(data["newlabel"] == type)))

data = pd.read_csv(path_meth, sep='\t').transpose()
data = data.reset_index()
headers = data.iloc[0]
data = pd.DataFrame(data.values[1:], columns=headers)
data_onlyvalues = data.drop(columns=["Composite Element REF"])
print("Meth")
print("shape :" + str(data.shape))
print("mean per feature" + str(data_onlyvalues.mean().sort_values(ascending=True)))
print("max per feature" + str(data_onlyvalues.max().sort_values(ascending=True)))
print("min per feature" + str(data_onlyvalues.min().sort_values()))

data = pd.read_csv("./data/Matrix_miRNA_deseq_correct.csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"] = "Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)
data_onlyvalues = data.drop(columns=["Composite Element REF"])
print("Matrix_miRNA_deseq_correct")
print("shape :" + str(data.shape))
print("mean per feature" + str(data_onlyvalues.mean().sort_values(ascending=True)))
print("max per feature" + str(data_onlyvalues.max().sort_values(ascending=True)))
print("min per feature" + str(data_onlyvalues.min().sort_values()))

data = pd.read_csv("./data/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"] = "Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)
data_onlyvalues = data.drop(columns=["Composite Element REF"])
print("Matrix_mRNA_deseq_normalized_prot_coding_correct")
print("shape :" + str(data.shape))
print("mean per feature" + str(data_onlyvalues.mean().sort_values(ascending=True)))
print("max per feature" + str(data_onlyvalues.max().sort_values(ascending=True)))
print("min per feature" + str(data_onlyvalues.min().sort_values()))
