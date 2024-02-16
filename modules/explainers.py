import json
import numpy as np
from modules.lib_IF import *
from modules.lib import *
from skopt.plots import partial_dependence_1D
from modules.ale import *

def ComputePDP(param_grid, model):

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)
        space.set_transformer_by_type('normalize',Categorical)
        space.set_transformer_by_type('normalize',Integer)
        space.set_transformer_by_type('normalize',Real)

        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
            
        pdp_samples = space.transform(space.rvs(n_samples=1000,random_state=123456))
        x_vals = list(param_grid.keys())

        n_dims = len(plot_dims)
        xi = []
        yi=[]
        for i in range(n_dims):
            index, dim = plot_dims[i]
            xi1, yi1 = partial_dependence_1D(space, model,
                                               index,
                                               samples=pdp_samples,
                                               n_points=40)
            xi.append(xi1)
            yi.append(yi1)
            
        x = [arr.tolist() for arr in xi]
        y = [arr for arr in yi]

        return x,y

def ComputePDP2D(param_grid, model,feature1,feature2):

        features = list(param_grid.keys())
        index1 = features.index(feature1)
        index2 = features.index(feature2)

         

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)
        space.set_transformer_by_type('normalize',Categorical)
        space.set_transformer_by_type('normalize',Integer)
        space.set_transformer_by_type('normalize',Real)

        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
        
        pdp_samples = space.transform(space.rvs(n_samples=1000,random_state=123456))

        _ ,dim_1 = plot_dims[index1]
        _ ,dim_2 = plot_dims[index2]
        xi, yi, zi = partial_dependence_2D(space, model,
                                                   index1, index2,
                                                   pdp_samples, 40)
        
        x = [arr.tolist() for arr in xi]
        y = [arr.tolist() for arr in yi]
        z = [arr.tolist() for arr in zi]
       
        return x,y,z
    
def ComputeALE(param_grid, model):

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)
        space.set_transformer_by_type('normalize',Categorical)
        space.set_transformer_by_type('normalize',Integer)
        space.set_transformer_by_type('normalize',Real)

        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
        print(param_grid)
        pdp_samples = space.transform(space.rvs(n_samples=1000,random_state=123456))
        data = pd.DataFrame(pdp_samples,columns=[n for n in name])
        x_vals = list(param_grid.keys())
        print(data)
        dataframes_list = []

        for i,feat in enumerate(name):
            ale_eff = compute_ALE(data,model,feat,space,pdp_samples,name,include_CI=False, C=0.95)
            dataframes_list.append(ale_eff)

        d = json.dumps([df.to_json(orient='split') for df in dataframes_list])

        return d