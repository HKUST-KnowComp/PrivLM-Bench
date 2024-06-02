
import numpy as np
import torch
from collections import defaultdict
from sklearn import metrics
import matplotlib.pyplot as plt
import os


run_name="p-tuning_qnli_bert-base_dp" 
directory = 'path_to_models'


def fpr_tpr(fprs, tprs,fpr_thres=1e-4):
    last_t=0.0
    for t,f in zip(tprs,fprs):
        if f>=fpr_thres:
            return t
        last_t=t

def plot_auc(x,y):
    # print(np.concatenate([np.ones(x.shape[0]),y.shape[0] ]).astype(int).shape ,np.concatenate( [x,y] ).shape )
    fpr, tpr, thresholds = metrics.roc_curve( np.concatenate([np.ones(x.shape[0]),np.zeros(y.shape[0])]).astype(int) ,np.concatenate( [x,y] ))
    
    auc=metrics.auc(fpr, tpr)
    log_tpr,log_fpr=np.log10(tpr),np.log10(fpr)
    log_tpr[log_tpr<-5]=-5
    log_fpr[log_fpr<-5]=-5
    log_fpr=(log_fpr+5)/5.0
    log_tpr=(log_tpr+5)/5.0
    log_auc=metrics.auc( log_fpr,log_tpr )
    fig=plt.figure(figsize=(4,4))
    plt.plot( fpr,tpr,label=f"(AUC = {auc:0.4f})")
    plt.plot( log_fpr,log_tpr,label=f" (Log AUC = {log_auc:0.4f})")
    plt.plot( np.array([0,1]),np.array([0,1]),label=f"baseline")
    plt.xlim([0,1.05])
    plt.ylim([0,1.05])
    plt.legend(loc=4)
    print("auc \t \t", auc)
    print("log auc \t",log_auc)
    print("tpr 1e-4 \t",fpr_tpr(fpr, tpr,1e-4))
    print("tpr 1e-2 \t",fpr_tpr(fpr, tpr,1e-2))
    print("tpr 1e-3 \t",fpr_tpr(fpr, tpr,1e-3))
    print("res!\t",auc, log_auc,fpr_tpr(fpr, tpr,1e-4),fpr_tpr(fpr, tpr,1e-3),fpr_tpr(fpr, tpr,1e-2))
    print(f"{auc:.4f},{log_auc:.4f},{fpr_tpr(fpr, tpr,1e-3):.4f},{fpr_tpr(fpr, tpr,1e-2):.4f}")
    # print(f"{auc},{log_auc},{fpr_tpr(fpr, tpr,1e-4)},")
    return auc,log_auc,fpr_tpr(fpr, tpr,1e-4),fpr_tpr(fpr, tpr,1e-3),fpr_tpr(fpr, tpr,1e-2)



import torch
import numpy as np
res=[]
for i in  range(130):
    try:
        if run_name=="prefix-tuning_qnli_bert-large_dp":
            p=torch.load(f"{directory}/qnli_bert-large-uncased_prefix-tuning_dp{i}.pt")
        if run_name=="p-tuning_qnli_roberta-base_dp":
            p=torch.load(f"{directory}/qnli_roberta-large_p-tuning_dp{i}.pt")
        if run_name=="p-tuning_qnli_roberta-base_non_dp":
            p=torch.load(f"{directory}/qnli_roberta-base_p-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_roberta-base_dp":
            p=torch.load(f"{directory}/qnli_roberta-base_prefix-tuning_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_bert-base_non-dp":
            p=torch.load(f"{directory}/qnli_bert-base-uncased_prefix-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_bert-base_dp":
            p=torch.load(f"{directory}/qnli_bert-base-uncased_prefix-tuning_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_roberta-large_dp":
            p=torch.load(f"{directory}/qnli_roberta-large_prefix-tuning_dp{i}.pt")
        if run_name=="p-tuning_qnli_roberta-large_non_dp":
            p=torch.load(f"{directory}/qnli_roberta-large_p-tuning_non_dp{i}.pt")
        if run_name=="p-tuning_qnli_bert-large_nondp":
            p=torch.load(f"{directory}/qnli_bert-large-uncased_p-tuning_non_dp{i}.pt")
        if run_name=="p-tuning_qnli_roberta-large_dp":
            p=torch.load(f"{directory}/qnli_roberta-large_p-tuning_dp{i}.pt")
        if run_name=="p-tuning_qnli_bert-large_dp":
            p=torch.load(f"{directory}/qnli_bert-large-uncased_p-tuning_dp{i}.pt")
        if run_name=="p-tuning_qnli_bert-base_dp":
            p=torch.load(f"{directory}/qnli_bert-base-uncased_p-tuning_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_bert-base_non_dp":
            p=torch.load(f"{directory}/qnli_bert-base-uncased_prefix-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_sst2_bert-base_non_dp":
            p=torch.load(f"{directory}/sst2_bert-base-uncased_prefix-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_sst2_roberta-base_non_dp":
            p=torch.load(f"{directory}/sst2_roberta-base_prefix-tuning_non_dp{i}.pt")
        if run_name=="sst2_bert-base_non_dp":
            p=torch.load(f"{directory}/sst2_bert-base-uncased_non_dp{i}.pt")
        if run_name=="sst2_bert-large_non_dp":
            p=torch.load(f"{directory}/sst2_bert-large-uncased_non_dp{i}.pt")
        if run_name=="qnil_bert-large_non_dp":
            p=torch.load(f"{directory}/qnli_bert-large-uncased_non_dp{i}.pt")
        if run_name=="qnli_bert-base_non_dp":
            p=torch.load(f"{directory}/qnli_bert-base-uncased_non_dp{i}.pt")
        if run_name=="sst2_bert-large_dp":
            p=torch.load(f"{directory}/sst2_bert-large-uncased_dp{i}.pt")
        if run_name=="qnli_bert-large_dp":
            p=torch.load(f"{directory}/qnli_bert-large-uncased_dp{i}.pt")
        if run_name=="qnli_roberta-base_dp":
            p=torch.load(f"{directory}/qnli_roberta-base_dp{i}.pt")
        if run_name=="qnil_roberta-large_dp":
            p=torch.load(f"{directory}/qnli_roberta-large_dp{i}.pt")
        if run_name=="sst2_large-base_dp":
            p=torch.load(f"{directory}/sst2_bert-large-uncased_dp{i}.pt")
        if run_name=="sst2_bert-base_dp":
            p=torch.load(f"{directory}/sst2_bert-base-uncased_dp{i}.pt")
        if run_name=="qnli_bert-base_dp":
            p=torch.load(f"{directory}/qnli_bert-base-uncased_dp{i}.pt")
        if run_name=="prefix-tuning_sst2_bert-large_non_dp":
            p=torch.load(f"{directory}/sst2_bert-large-uncased_prefix-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_bert-large_non_dp":
            p=torch.load(f"{directory}/qnli_bert-large-uncased_prefix-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_roberta-base_non_dp":
            p=torch.load(f"{directory}/qnli_roberta-base_prefix-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_sst2_roberta-large_non_dp":
            p=torch.load(f"{directory}/sst2_roberta-large_prefix-tuning_non_dp{i}.pt")
        if run_name=="prefix-tuning_qnli_roberta-large_non_dp":
            p=torch.load(f"{directory}/qnli_roberta-large_prefix-tuning_non_dp{i}.pt")
        if run_name=="p-tuning_sst2_roberta-base_non_dp":
            p=torch.load(f"{directory}/sst2_roberta-base_prefix-tuning_non_dp{i}.pt")
        if run_name=="p-tuning_sst2_roberta-base_dp":
            p=torch.load(f"{directory}/sst2_roberta-base_prefix-tuning_dp{i}.pt")
        if run_name=="p-tuning_qnil_roberta-base_non_dp":
            p=torch.load(f"{directory}/sst2_roberta-base_p-tuning_dp{i}.pt")
        if run_name=="prefix_qnil_roberta-base_non_dp":
            p=torch.load(f"{directory}/qnli_roberta-base_prefix-tuning_dp{i}.pt")
        if run_name=="mnli_roberta-base_non_dp":
            p=torch.load(f"{directory}/mnli_roberta-base_non_dp{i}.pt")
        if run_name=="mnli_roberta-large_dp":
            p=torch.load(f"{directory}/mnli_roberta-large_dp{i}.pt")
        if run_name=="sst2_roberta-large_non_dp":
            p=torch.load(f"{directory}/sst2_roberta-large_non_dp{i}.pt")
        if run_name=="sst2_roberta-base_non_dp":
            p=torch.load(f"{directory}/sst2_roberta-base_non_dp{i}.pt")
        if run_name=="sst2_roberta-base_dp":
            p=torch.load(f"{directory}/sst2_roberta-base_dp{i}.pt")
        if run_name=="qnil_roberta-large_non_dp":
            p=torch.load(f"{directory}/qnli_roberta-large_non_dp{i}.pt")
        if run_name=="qnli_roberta-base_non_dp":
            p=torch.load(f"{directory}/qnli_roberta-base_non_dp{i}.pt")
    except FileNotFoundError:
        continue
    print(i,p["acc"])
    # print(p["probs"].shape )
    try:
        p["probs"]=np.concatenate(p["probs"])
        p["labels"]=np.concatenate(p["labels"])
    except ValueError:
        pass

    res.append(p)


test_res=res[0]
res=res[1:]
avg_acc=np.array([a["acc"] for a in res]).mean()
print("avg acc",avg_acc)

aucs=[]
train_prob_dict={}
test_prob_dict={}
count=0
for a in res:
    try:
        tmp=a["probs"][ np.array(list(range(len(a["labels"])))),:]
    except IndexError:
        continue
    # print(count)
    count+=1
    
    try :
        prob=tmp[a["labels"]].numpy()
    except :
        prob=tmp[a["labels"]]
    
    index_array=[]
    for i in range(a["labels"].max()+1):
        # print(i)
        index_array.append( (a["labels"]!=i).reshape(-1,1) )
    index_array=np.hstack(index_array).astype(float)
    # print(tmp[index_array])
    prob_=(tmp*index_array).sum(axis=1)
    # print(prob_)
    # tmp[:,a["labels"]]=0
    # prob_=tmp.sum(axis=1)
    # prob_=1-prob
    # prob_=a["probs"][ np.array(list(range(len(a["labels"])))),1-a["labels"]]
    for i in a["train_idx"]:
        if i>=len(prob):
            continue
        if i not in train_prob_dict:
            train_prob_dict[i]=[(prob[i]/(1e-6+prob_[i]))]
        else:
            train_prob_dict[i].append((prob[i]/(1e-6+prob_[i])))
    for i in a["test_idx"]:
        if i>=len(prob):
            continue
        if i not in test_prob_dict:
            test_prob_dict[i]=[(prob[i]/(1e-6+prob_[i]))]
        else:
            test_prob_dict[i].append((prob[i]/(1e-6+prob_[i])))


total=len(train_prob_dict)
prob_train_mean=np.array([np.array(train_prob_dict[i]).mean() for i in range(total)])
prob_test_mean=np.array([np.array(test_prob_dict[i]).mean() for i in range(total)])
prob_train_var=np.array([np.array(train_prob_dict[i]).std() for i in range(total)])
prob_test_var=np.array([np.array(test_prob_dict[i]).std() for i in range(total)])


x=test_res["probs"][ np.array(list(range(len(test_res["labels"])))),test_res["labels"]]
x=x/(1e-6+1.0-x)
y=np.zeros(len(x))
y[test_res["train_idx"]]=1
from scipy.stats import norm

prob_in=norm.pdf(x, prob_train_mean, prob_train_var )
prob_out=norm.pdf(x, prob_test_mean, prob_test_var )

likelihood_score=np.log(prob_in/(prob_out+1e-20))
likelihood_score[np.isinf(likelihood_score)] = np.max(likelihood_score[np.isfinite(likelihood_score)])
# Replace -inf values with the minimum value
likelihood_score[np.isinf(likelihood_score)] = np.min(likelihood_score[np.isfinite(likelihood_score)])
bins=np.linspace(-10,10,100)
plt.hist(likelihood_score[test_res["train_idx"]],bins=bins,alpha=0.5)
plt.hist(likelihood_score[test_res["test_idx"]],bins=bins,alpha=0.5)
plt.yscale("log")
res=plot_auc(likelihood_score[test_res["train_idx"]],likelihood_score[test_res["test_idx"]])
print(res)