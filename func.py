import gurobipy as gp
import numpy as np
import os

if not os.path.exist('./generated'):
    os.mkdir('./generated')



nCol = 1000

def read_A(fnm='./zib03.mps.gz'):
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

def get_block(A,nCol,bounds,c,niter=0):
    A = A[:,niter*nCol:niter*nCol+nCol]
    bounds = bounds[niter*nCol:niter*nCol+nCol]
    c = c[niter*nCol:niter*nCol+nCol]
    m = A.shape[0]
    n = A.shape[1]

    x = np.random.rand(n)
    
    # print(x)
    # print(bounds)
    
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


    indx = []
    vals = []

    for iRow in range(ptr.shape[0]-1):
        # iIndx~iIndx+1
        flag=False
        tmp_row = []
        tmp_val = []
        for iPtr in range(ptr[iRow],ptr[iRow+1]):
            iCol = indx_ori[iPtr]
            # indx[0].append(iRow)
            # indx[1].append(iCol)
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


def gen_mps(indx,vals,m,n,rhs,lb,ub,c,nins=0):
    model = gp.Model('zib')
    vs = model.addVars(n, lb=lb, ub=ub, obj=c)
    model.update()
    
    model.addConstrs(gp.quicksum(vs[indx[i][j]]*vals[i][j] for j in range(len(indx[i]))) <= rhs[i] for i in range(m))

    # model.optimize()
    
    model.write(f'./generated/{nins}.mps')
    



A,bounds, b, c = read_A()  

total_iter = A.shape[1]//nCol
residual = A.shape[1] % nCol

for i in range(total_iter+1):
    print(f'{i}th iteration (out of {total_iter})')
    indx,vals,m,n,local_rhs,lb,ub,new_A,local_c = get_block(A,nCol,bounds,c,niter=i)
    gen_mps( indx,vals,m,n,local_rhs,lb,ub,local_c,nins=i)
    
    