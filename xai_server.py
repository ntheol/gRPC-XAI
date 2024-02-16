import grpc
from concurrent import futures
import xai_service_pb2_grpc
import xai_service_pb2
from xai_service_pb2_grpc import ExplanationsServicer
import json
import numpy as np
import torch
import io
from concurrent import futures
from modules.lib_IF import *
import torch.nn.functional as F
import json
from sklearn.inspection import partial_dependence
from modules.lib import *
from modules.ale import *
from modules.explainers import *
import dice_ml

class MyExplanationsService(ExplanationsServicer):

    def GetExplanation(self, request_iterator, context):
        print('Reading data')
        dataframe = pd.DataFrame()
        label = pd.DataFrame()
        for request in request_iterator:
            explanation_type = request.explanation_type
            explanation_method = request.explanation_method

            if explanation_type == 'Pipeline':

                if explanation_method == 'PDPlots':

                    param_grid = json.loads(request.param_grid)
                    model = torch.load(io.BytesIO(request.model))  
                    x,y = ComputePDP(param_grid=param_grid, model=model)

                    return xai_service_pb2.ExplanationsResponse(
                        pdp_hp_values=json.dumps(x),
                        pdp_values=json.dumps(y)
                    )

                elif explanation_method == '2D_PDPlots':

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

                elif explanation_method == 'ALEPlots':

                    param_grid = json.loads(request.param_grid)
                    print(param_grid)
                    model = torch.load(io.BytesIO(request.model)) 
                    d = ComputeALE(param_grid=param_grid, model=model)

                    return xai_service_pb2.ExplanationsResponse(
                        ale_data=d  # Replace with actual data
                    )

                elif explanation_method == 'InfluenceFunctions':  
                    print("receiving")
                    chunk_df = pd.read_parquet(io.BytesIO(request.train_data))  # Deserialize DataFrame chunk
                    dataframe = pd.concat([dataframe, chunk_df], ignore_index=True)


                    chunk_df_label = pd.read_parquet(io.BytesIO(request.train_labels))  # Deserialize DataFrame chunk
                    label = pd.concat([label, chunk_df_label], ignore_index=True)

                elif explanation_method == 'CounterfactualExplanations':  
                    print("receiving")
                    proxy_dataset = pd.read_parquet(io.BytesIO(request.proxy_dataset))  # Deserialize DataFrame chunk
                    query = pd.read_parquet(io.BytesIO(request.query))
                    model = torch.load(io.BytesIO(request.model), map_location=torch.device('cpu'))  

                    d = dice_ml.Data(dataframe=proxy_dataset, 
                        continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist()
                        , outcome_name='BinaryLabel')
                    
                    # Using sklearn backend
                    m = dice_ml.Model(model=model, backend="sklearn")
                    # Using method=random for generating CFs
                    exp = dice_ml.Dice(d, m, method="random")
                    e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)
                    e1.visualize_as_dataframe(show_only_changes=True)
                    cfs = e1.cf_examples_list[0].final_cfs_df
                    display(cfs)
                    cfs = cfs.to_parquet(None)

                    return xai_service_pb2.ExplanationsResponse(
                        cfs=cfs
                    )
            elif explanation_type == 'Model':
                if explanation_method == 'PDPlots' or explanation_method == '2D_PDPlots':
                    chunk_df = pd.read_parquet(io.BytesIO(request.train_data))
                    dataframe = pd.concat([dataframe, chunk_df], ignore_index=True)


        if explanation_method == 'PDPlots' and explanation_type == 'Model':
            model = torch.load(io.BytesIO(request.model), map_location=torch.device('cpu')) 
            features = request.features
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
            categorical_features = dataframe.columns.drop(numeric_features)

            pdp = partial_dependence(model, dataframe, features = [dataframe.columns.tolist().index(features)],
                                     feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
            
            return xai_service_pb2.ExplanationsResponse(
                        pdp_vals=json.dumps([value.tolist() for value in pdp['grid_values']]),
                        pdp_effect=json.dumps(pdp['average'].tolist())
                    )
        
        elif explanation_method == '2D_PDPlots' and explanation_type == 'Model':
            model = torch.load(io.BytesIO(request.model), map_location=torch.device('cpu')) 
            feature1 = request.feature1
            feature2 = request.feature2
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
            categorical_features = dataframe.columns.drop(numeric_features)

            pdp = partial_dependence(model, dataframe, features = [(dataframe.columns.tolist().index(feature1),dataframe.columns.tolist().index(feature2))],
                                     feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
            
            return xai_service_pb2.ExplanationsResponse(
                        pdp_vals=json.dumps([value.tolist() for value in pdp['grid_values']]),
                        pdp_effect=json.dumps(pdp['average'].tolist())
                    )

        elif explanation_method == 'InfluenceFunctions':
            model = torch.load(io.BytesIO(request.model), map_location=torch.device('cpu'))             
            test_data =  pd.read_parquet(io.BytesIO(request.test_data))        
            test_labels =  pd.read_parquet(io.BytesIO(request.test_labels))
            num_influential = request.num_influential
            print('Data received')
            print(len(dataframe))
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
            response = xai_service_pb2.ExplanationsResponse(influences=influences,positive=positive,negative = negative)

            return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(MyExplanationsService(), server)
    #xai_service_pb2_grpc.add_InfluencesServicer_to_server(MyInfluencesService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
