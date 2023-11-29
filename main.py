import pandas as pd
import torch
import numpy as np
from lib_IF import *
import grpc
import influence_service_pb2
import influence_service_pb2_grpc
import io
from support.torch import (
    TorchMLP
)
def compute_influences_via_grpc(model, train_data, test_data, train_labels,test_labels,num_influential, untransformed_train, untransformed_train_labels):
    channel = grpc.insecure_channel('localhost:50051')
    stub = influence_service_pb2_grpc.InfluenceServiceStub(channel)

    # Serialize the PyTorch model and data
    model_bytes = io.BytesIO()
    torch.save(model, model_bytes)
    
    request = influence_service_pb2.InfluenceRequest(
        model=model_bytes.getvalue(),
        train_data=train_data.to_parquet(None),
        test_data=test_data.to_parquet(None),
        train_labels = train_labels.to_parquet(None),
        test_labels = test_labels.to_parquet(None),
        num_influential = num_influential,
        untransformed_train = untransformed_train.to_parquet(None),
        untransformed_train_labels = untransformed_train_labels.to_parquet(None)
    )

    response = stub.ComputeInfluences(request)
    return response.influences, response.positive, response.negative

def main():

    #Load data
    train = pd.read_csv('train.csv').drop(columns='Unnamed: 0')
    test = pd.read_csv('test.csv').drop(columns='Unnamed: 0')
    transformed_train = pd.read_csv('transformed_train.csv').drop(columns='Unnamed: 0')
    transformed_test = pd.read_csv('transformed_test.csv').drop(columns='Unnamed: 0')
    train_labels = pd.read_csv('trainn_labels.csv').drop(columns='Unnamed: 0')
    test_labels = pd.read_csv('test_labels.csv').drop(columns='Unnamed: 0')



    #Load pytorch model
    feature_dimension = 47
    num_classes = 1
    network_size = [32,32,32]
    layers_size = [feature_dimension, *network_size, num_classes]
    new = TorchMLP(layers_size)
    new.load_state_dict(torch.load("model.pth"))

    influences,positive,negative = compute_influences_via_grpc(new, transformed_train.loc[0:2000], transformed_test.loc[0:2000], train_labels.loc[0:2000], test_labels.loc[0:2000],5, train.loc[0:2000], test_labels.loc[0:2000])
    inf = pd.DataFrame(np.array(influences).reshape(1000,1000),columns=[str(i) for i in range(0, 1000)])
    positive =  pd.read_parquet(io.BytesIO(positive))
    negative =  pd.read_parquet(io.BytesIO(negative))





if __name__ == "__main__":
    main()

