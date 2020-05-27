
import pandas as pd
path_annotation="./data/annotation_global.csv"
path_meth="./data/Matrix_meth.csv"


def union(row):
    return str(row["is_tumor"])+"-"+row["project_id"]
def map(row):
    return str(row["case_id"])+"_"+str(row["is_tumor"])

data = pd.read_csv(path_annotation, sep='\t')


data["case_id"]=data.apply(lambda row: map(row),axis=1)
cases_removed=data[data["project_id"]=="TCGA-SARC"]
data= data[data["project_id"]!="TCGA-SARC"]
data["is_tumor"]=data["is_tumor"].map({0: 'sane', 1: 'tumor'})
data["label"]=data.apply(lambda row: union(row),axis=1)
data=data.sort_values(by="case_id")
#data=data.drop(columns=["is_tumor","project_id"])
data.to_csv("./data/preprocessed_annotation_global.csv")


data = pd.read_csv(path_meth, sep='\t').transpose()
data = data.reset_index()
headers = data.iloc[0]
data = pd.DataFrame(data.values[1:], columns=headers)

index=data["Composite Element REF"]
data_onlyvalues=data.drop(columns=["Composite Element REF"])
data=data[data_onlyvalues.std().sort_values(ascending=False).head(10000).index]
data["Composite Element REF"]=index

data= data[~data["Composite Element REF"].isin(cases_removed["case_id"]) ]
data=data.sort_values(by="Composite Element REF")
data.to_csv("./data/preprocessed_Matrix_meth.csv")

data= pd.read_csv("./data/Matrix_miRNA_deseq_correct.csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"]="Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)

index=data["Composite Element REF"]
data_onlyvalues=data.drop(columns=["Composite Element REF"])
data=data[data_onlyvalues.std().sort_values(ascending=False).head(10000).index]
data["Composite Element REF"]=index

data= data[~data["Composite Element REF"].isin(cases_removed["case_id"]) ]
data=data.sort_values(by="Composite Element REF")
data.to_csv("./data/preprocessed_Matrix_miRNA_deseq_correct.csv")




data= pd.read_csv("./data/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"]="Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)

index=data["Composite Element REF"]
data_onlyvalues=data.drop(columns=["Composite Element REF"])
data=data[data_onlyvalues.std().sort_values(ascending=False).head(10000).index]
data["Composite Element REF"]=index

data= data[~data["Composite Element REF"].isin(cases_removed["case_id"]) ]
data=data.sort_values(by="Composite Element REF")
data.to_csv("./data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv")



