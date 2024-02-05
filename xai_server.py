import grpc
from concurrent import futures
import xai_service_pb2_grpc
import xai_service_pb2
from xai_service_pb2_grpc import ExplanationsServicer,InfluencesServicer
import json
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
from modules.explainers import *

class MyExplanationsService(ExplanationsServicer):

    def GetExplanation(self, request, context):
        explanation_type = request.explanation_type

        if explanation_type == 'ComputePDP':
            param_grid = json.loads(request.param_grid)
            model = torch.load(io.BytesIO(request.model))  
            x,y = ComputePDP(param_grid=param_grid, model=model)
            return xai_service_pb2.ExplanationsResponse(
                pdp_hp_values=json.dumps(x),
                pdp_values=json.dumps(y)
            )

        elif explanation_type == 'ComputePDP2D':
            param_grid = json.loads(request.param_grid)
            model = torch.load(io.BytesIO(request.model))  
            feature1 = request.feature1
            feature2 = request.feature2
            x,y,z = ComputePDP2D(param_grid=param_grid, model=model,feature1=feature1,feature2=feature2)
            return xai_service_pb2.ExplanationsResponse(
                pdp2d_xi=json.dumps(x),
                pdp2d_yi=json.dumps(y),
                pdp2d_zi=json.dumps(z)
            )

        elif explanation_type == 'ComputeALE':
            param_grid = json.loads(request.param_grid)
            model = torch.load(io.BytesIO(request.model)) 
            d = ComputeALE(param_grid=param_grid, model=model)
            return xai_service_pb2.ExplanationsResponse(
                ale_data=d  # Replace with actual data
            )

class MyInfluencesService(InfluencesServicer):
    
    def ComputeInfluences(self, request_iterator, context):
        # Deserialize the model and data
        print('Reading data')
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
        print('Data received')
        # untransformed_train = pd.read_parquet(io.BytesIO(request.untransformed_train))
        # untransformed_train_labels = pd.read_parquet(io.BytesIO(request.untransformed_train_labels))

        y = label.squeeze().to_numpy()
        yt = test_labels.squeeze().to_numpy()

        x = dataframe.values
        xt = test_data.values
        
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
        response = xai_service_pb2.InfluenceResponse(influences=influences,positive=positive,negative = negative)
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(MyExplanationsService(), server)
    xai_service_pb2_grpc.add_InfluencesServicer_to_server(MyInfluencesService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
