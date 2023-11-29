import grpc
import influence_service_copy_pb2
import influence_service_copy_pb2_grpc
import numpy as np
import torch
import io
from concurrent import futures
from modules.lib_IF import *
import torch.nn.functional as F
import json,joblib
from modules.lib import *
from skopt.plots import partial_dependence_1D
from modules.ale import *

class InfluenceServiceCopy(influence_service_copy_pb2_grpc.InfluenceServiceCopyServicer):
    
    def ComputeInfluences(self, request_iterator, context):
        # Deserialize the model and data

        dataframe = pd.DataFrame()
        label = pd.DataFrame()
        for request in request_iterator:
            chunk_df = pd.read_parquet(io.BytesIO(request.train_data))  # Deserialize DataFrame chunk
            dataframe = pd.concat([dataframe, chunk_df], ignore_index=True)


            chunk_df_label = pd.read_parquet(io.BytesIO(request.train_labels))  # Deserialize DataFrame chunk
            label = pd.concat([label, chunk_df_label], ignore_index=True)
            
        model = torch.load(io.BytesIO(request.model), map_location=torch.device('cpu'))             
        test_data =  pd.read_parquet(io.BytesIO(request.test_data))        
        test_labels =  pd.read_parquet(io.BytesIO(request.test_labels))
        num_influential = request.num_influential
        untransformed_train = pd.read_parquet(io.BytesIO(request.untransformed_train))
        untransformed_train_labels = pd.read_parquet(io.BytesIO(request.untransformed_train_labels))

        num_train=1000
        num_test=1000

        y = label.squeeze().to_numpy()
        yt = test_labels.squeeze().to_numpy()[:num_test]

        x = dataframe.values
        xt = test_data.values[:num_test]
  

        #Compute influences 
        influences = compute_IF(model=model,loss=F.binary_cross_entropy,training_data = x, 
                                test_data = xt, train_labels= y, test_labels= yt,
                                influence_type='up', inversion_method='direct', hessian_regularization=0.5)

        show_influences(influences,5,dataframe,label)
        positive = show_pos_inf_instance(influences,num_influential,dataframe,label)
        negative = show_neg_inf_instance(influences,num_influential,dataframe,label)



        influences = influences.flatten().tolist()
        positive = positive.to_parquet(None)
        negative = negative.to_parquet(None)
        # Create a response message
        response = influence_service_copy_pb2.InfluenceResponse(influences=influences,positive=positive,negative = negative)
        return response
    
    def ComputePDP(self, request, context):

        param_grid = json.loads(request.param_grid)
        model = torch.load(io.BytesIO(request.model))   

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)
        space.set_transformer_by_type('normalize',Categorical)

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

        response = influence_service_copy_pb2.PDPResponse(hp_values=json.dumps(x),pdp_values=json.dumps(y))
        return response

    def ComputePDP2D(self, request, context):

        param_grid = json.loads(request.param_grid)
        model = torch.load(io.BytesIO(request.model))  
        feature1 = request.feature1
        feature2 = request.feature2

        features = list(param_grid.keys())
        index1 = features.index(feature1)
        index2 = features.index(feature2)

         

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)
        space.set_transformer_by_type('normalize',Categorical)

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

        response = influence_service_copy_pb2.PDP2DResponse(xi=json.dumps(x),yi=json.dumps(y),zi=json.dumps(z))        
        return response
    
    def ComputeALE(self, request, context):

        data = pd.read_parquet(io.BytesIO(request.data)) 
        param_grid = json.loads(request.param_grid)
        model = torch.load(io.BytesIO(request.model))  

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)
        space.set_transformer_by_type('normalize',Categorical)

        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
        
        pdp_samples = space.transform(space.rvs(n_samples=1000,random_state=123456))

        dataframes_list = []

        for i,feat in enumerate(name):
            ale_eff = compute_ALE(data,model,feat,space,pdp_samples,name,include_CI=False, C=0.95)
            dataframes_list.append(ale_eff)

        d = json.dumps([df.to_json(orient='split') for df in dataframes_list])
        response = influence_service_copy_pb2.ALEResponse(data=d)    

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    influence_service_copy_pb2_grpc.add_InfluenceServiceCopyServicer_to_server(InfluenceServiceCopy(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
