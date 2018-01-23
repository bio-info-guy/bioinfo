import sys
import numpy as np

class NeoKmeans:
    def init_center(self): #K-means ++ initialization
        centers=[]
        matrix_cp=np.copy(self.matrix)
        first_pt=np.random.choice(self.n, 1, replace=False)
        init_point=matrix_cp[first_pt,:]
        centers.append(init_point)
#        matrix_cp=np.delete(matrix_cp, first_pt, axis=1)
        dists=[]
        while len(centers) < self.k:
            new_dist=np.nansum(np.square(np.array(centers[-1])-matrix_cp), axis=1)
            dists.append(new_dist)
            probs=np.square(np.amin(np.transpose(np.array(dists)), axis=1))/np.sum(np.square(np.amin(np.transpose(np.array(dists)), axis=1)))
            new_point=np.random.choice(self.n, 1, p=probs)
            centers.append(matrix_cp[new_point,:])
        return np.vstack(np.array(centers))

    def k_means(self, neo=True):
        class_matrix=np.zeros(shape=(self.n, self.k))
        centers=self.init_center()
        centers_list=[[] for i in range(self.k)]
        iter=0; update=2
        while iter < self.niter and update >= self.miu:
            self.old_centers=centers
            dist=np.nansum(np.square((centers-self.matrix[:,np.newaxis])), axis=2)
            index=np.argmin(dist, axis=1)
            print(index)
            if neo:
                cOne = round(self.n-self.n*self.beta)
                for i in range(cOne):
                    j=index[i]
                    centers_list[j].append(self.matrix[i,:])
                    dist[i,j]=0
                    class_matrix[i,j]=1
                cTwo = round(self.n*self.alpha+self.n*self.beta)
                constraint_a=np.unravel_index(np.argpartition(dist.flatten(), cTwo)[:cTwo], dist.shape) 
                for ind in range(len(constraint_a[0])):
                    i=constraint_a[0][ind]
                    j=constraint_a[1][ind]
                    centers_list[j].append(self.matrix[i,:])
                    class_matrix[i,j]=1
            else:
                for i in range(self.n):
                    j=index[i]
                    centers_list[j].append(self.matrix[i:])
                    class_matrix[i,j]=1
            for num in range(self.k):
                if centers_list[num] == []:
                    centers_list[num]=[centers[num]]
            for i in range(len(centers_list)):
                centers_list[i]=np.mean(np.vstack(centers_list[i]), axis=0)
            update=np.mean(np.square(centers-self.old_centers))
            iter+=1
        return class_matrix, dist

    def findParam(self):
        pass

    def __init__(self, matrix, nclust=3, alpha=0.1, beta=0.8, niteration=10, miu=1, sim_matrix=False):
        self.k=nclust
        self.n=matrix.shape[0]
        self.alpha=alpha
        self.beta=beta
        self.miu=miu
        self.niter=niteration
        if sim_matrix==False:
            self.matrix=np.cov(matrix)
        else:
            self.matrix=matrix

    
