import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('circuits_complete.csv')
META=['circuit','Error_Cairo','Error_Hanoi','num_qubits','depth']
G=[c for c in df.columns if c not in META]
two=[g for g in ["cx","cz","cy","ch","cs","csdg","csx","cp","crx","cry","crz","cu","cu1","cu3","swap","iswap","dcx","ecr","rxx","ryy","rzz","rzx","xx_minus_yy","xx_plus_yy"] if g in df]
three=[g for g in ["ccx","ccz","cswap","rccx","c3sx","rcccx"] if g in df]
non=[g for g in ["measure","reset","barrier"] if g in df]
one=[g for g in G if g not in two+three+non]
df['total_gates']=df[one+two+three].sum(axis=1); df['n_1q']=df[one].sum(axis=1); df['n_2q']=df[two].sum(axis=1); df['n_3q_plus']=df[three].sum(axis=1)
df['ratio_2q']=(df.n_2q+df.n_3q_plus)/df.total_gates.clip(lower=1); df['gates_per_layer']=df.total_gates/df.depth.clip(lower=1); df['gates_per_qubit']=df.total_gates/df.num_qubits
FS={'A':G,'D':G+['num_qubits','depth','total_gates','n_1q','n_2q','n_3q_plus','ratio_2q','gates_per_layer','gates_per_qubit']}
rows=[]
for b in ['Cairo','Hanoi']:
    y=df['Error_'+b].values
    for frac in [0.01,0.8]:
        for fs,cols in FS.items():
            X=df[cols].values; m=[];r=[];t=[]
            for seed in [0,1,2]:
                tr,te=train_test_split(np.arange(len(df)),train_size=frac,random_state=seed)
                mdl=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(80,40),learning_rate_init=1e-3,max_iter=300,early_stopping=(frac>0.05),random_state=seed))
                t0=time.time(); mdl.fit(X[tr],y[tr]); t.append(time.time()-t0)
                p=mdl.predict(X[te]); m.append(mean_absolute_error(y[te],p)); r.append(mean_squared_error(y[te],p)**.5)
            rows.append(dict(backend=b,train_frac=frac,features=fs,MAE=round(np.mean(m),4),MAE_std=round(np.std(m),4),RMSE=round(np.mean(r),4),fit_s=round(np.mean(t),1)))
            print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('results_dense_nn.csv',index=False)
