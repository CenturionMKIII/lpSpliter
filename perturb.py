import gurobipy as gp
import os
import random
import numpy as np
random.seed(0)



def read_A(fnm='./cont1.mps'):
    model = gp.read(fnm)
    A = model.getA()
    bounds = []
    cost = []
    for v in model.getVars():
        bounds.append([v.LB,v.UB])
        cost.append(v.Obj)
    cs = model.getConstrs()
    b = []
    for c in cs:
        b.append(c.RHS)
    return A, bounds, b, cost


if not os.path.exists('./generated'):
    os.mkdir('./generated')



class norm_dist_pert:
    def __init__(self,ratio=0.1):
        self.ratio = ratio

    def run(self,ori):
        for i in range(len(ori)):
            ori[i] = ori[i] * (1.0 + random.random()*2.0*self.ratio - self.ratio)
        return  ori



def get_prob(A,bounds,c, cons_pert_func=None, cost_pert_func=None):
    m = A.shape[0]
    n = A.shape[1]

    if cost_pert_func is not None:
        c = cost_pert_func.run(c)

    x = np.random.rand(n)
    
    max_bound = 1e+4
    
    for i in range(n):
        if bounds[i][1] > max_bound:
            bounds[i][1] = max_bound
        if bounds[i][0] < -max_bound:
            bounds[i][0] = -max_bound
            
        intv = bounds[i][1] - bounds[i][0]
        x2 = x[i]*intv + bounds[i][0]
        # print(f'{x[i]} -> {x2},  {intv}')
        # input()
    
    
    rhs = A*x
    
    real_rhs = []
    
    indx_ori = A.indices
    ptr = A.indptr
    valsv = A.data
    if cons_pert_func is not None:
        valsv = cons_pert_func.run(valsv)


    indx = []
    vals = []

    for iRow in range(ptr.shape[0]-1):
        # iIndx~iIndx+1
        flag=False
        tmp_row = []
        tmp_val = []
        for iPtr in range(ptr[iRow],ptr[iRow+1]):
            iCol = indx_ori[iPtr]
            tmp_row.append(iCol)
                
            tmp_val.append(valsv[iPtr])

            flag=True
        if flag:
            real_rhs.append(rhs[iRow])
            indx.append(tmp_row)
            vals.append(tmp_val)
        
    # indx=np.array(indx)   
    # vals=np.array(vals)    
    real_rhs = np.array(real_rhs)
    m = real_rhs.shape[0]
    
    bounds = np.array(bounds)
    lb = bounds[:,0]
    ub = bounds[:,1]
    
    
    return indx,vals,m,n,real_rhs,lb,ub,A,c


def gen_mps(indx,vals,m,n,rhs,lb,ub,c,ori_filname='./cont1.mps'):
    ori_filname = ori_filname.replace('./','').replace('.mps','').replace('.gz','')
    model = gp.Model('zib')
    vs = model.addVars(n, lb=lb, ub=ub, obj=c)
    model.update()
    
    model.addConstrs(gp.quicksum(vs[indx[i][j]]*vals[i][j] for j in range(len(indx[i]))) <= rhs[i] for i in range(m))

    # model.optimize()
    
    model.write(f'./generated/{ori_filname}.mps.gz')





fnm = './cont1.mps'
A,bounds, b, c = read_A(fnm)  

cons_perturber=norm_dist_pert(0.2)
cost_perturber=norm_dist_pert(0.2)

indx,vals,m,n,local_rhs,lb,ub,new_A,local_c = get_prob(A,bounds,c,cons_pert_func=cons_perturber, cost_pert_func=cost_perturber)
gen_mps(indx,vals,m,n,local_rhs,lb,ub,local_c,ori_filname=fnm)