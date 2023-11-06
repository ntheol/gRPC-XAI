import grpc
import influence_service_pb2
import influence_service_pb2_grpc
import numpy as np
import torch
import io
from concurrent import futures
from modules.lib_IF import *
import torch.nn.functional as F
from numproto import ndarray_to_proto, proto_to_ndarray


class InfluenceService(influence_service_pb2_grpc.InfluenceServiceServicer):
    
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
        response = influence_service_pb2.InfluenceResponse(influences=influences,positive=positive,negative = negative)
        return response
    

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    influence_service_pb2_grpc.add_InfluenceServiceServicer_to_server(InfluenceService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
