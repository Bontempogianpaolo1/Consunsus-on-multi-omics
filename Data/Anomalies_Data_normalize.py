from scipy import stats
import numpy as np
import pandas as pd
path_annotation="./original/anomalies_annotation_global.csv"
path_meth="./original/anomalies_Matrix_meth.csv"
max_features=5000
filenames = ["meth","mrna","micro mrna"]
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
data.to_csv("./data/anomalies_preprocessed_annotation_global.csv",index=False)

matrixname="meth"
data = pd.read_csv("./original/anomalies_Matrix_"+matrixname+".csv", sep='\t').transpose()
data = data.reset_index()
headers = data.iloc[0]
data = pd.DataFrame(data.values[1:], columns=headers)
index=data["Composite Element REF"]
data_onlyvalues=data.drop(columns=["Composite Element REF"])
data2=pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
data2["Composite Element REF"]=index
data2=data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data2.to_csv("./data/anomalies_Matrix_"+matrixname+".csv",index=False,header=False)
#data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
data=pd.DataFrame(stats.zscore(data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index].values.tolist()))
data["Composite Element REF"]=index
data= data[~data["Composite Element REF"].isin(cases_removed["case_id"]) ]
data=data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data.to_csv("./data/anomalies_preprocessed_Matrix_"+matrixname+".csv",index=False,header=False)


matrixname="miRNA_deseq_correct"
data = pd.read_csv("./original/anomalies_Matrix_"+matrixname+".csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"]="Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)
index=data["Composite Element REF"]
data_onlyvalues=data.drop(columns=["Composite Element REF"])
data2=pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
data2["Composite Element REF"]=index
data2=data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data2.to_csv("./data/anomalies_Matrix_"+matrixname+".csv",index=False,header=False)
#data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
data=pd.DataFrame(stats.zscore(data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index].values.tolist()))
data["Composite Element REF"]=index
data= data[~data["Composite Element REF"].isin(cases_removed["case_id"]) ]
matrixname=filenames[1]
data=data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data.to_csv("./data/anomalies_preprocessed_Matrix_"+matrixname+".csv",index=False,header=False)



matrixname="mRNA_deseq_normalized_prot_coding_correct"
data = pd.read_csv("./original/anomalies_Matrix_"+matrixname+".csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"]="Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)
index=data["Composite Element REF"]
data_onlyvalues=data.drop(columns=["Composite Element REF"])
data2=pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
data2["Composite Element REF"]=index
data2=data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data2.to_csv("./data/anomalies_Matrix_"+matrixname+".csv",index=False,header=False)
#data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
data=pd.DataFrame(stats.zscore(data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index].values.tolist()))
data["Composite Element REF"]=index
data= data[~data["Composite Element REF"].isin(cases_removed["case_id"]) ]
matrixname=filenames[2]
data=data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data.to_csv("./data/anomalies_preprocessed_Matrix_"+matrixname+".csv",index=False,header=False)



