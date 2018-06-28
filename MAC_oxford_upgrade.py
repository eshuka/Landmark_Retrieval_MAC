import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
from scipy import spatial


def get_top_k_result(similar_list, k):
    result = (sorted(similar_list, key=lambda l: l[1], reverse=True))
    return result[1:k + 1]


def layer_feature(input_img, model):
    layer0_feature = model.conv1.forward(input_img)
    layer0_feature = model.bn1.forward(layer0_feature)
    layer0_feature = model.relu.forward(layer0_feature)
    layer0_feature = model.maxpool.forward(layer0_feature)
    layer1_feature = model.layer1.forward(layer0_feature)
    layer2_feature = model.layer2.forward(layer1_feature)
    layer3_feature = model.layer3.forward(layer2_feature)
    layer4_feature = model.layer4.forward(layer3_feature)

    return layer1_feature, layer2_feature, layer3_feature, layer4_feature


def query_feature(query_image, model):
    centre_crop = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    query_img = Image.open(query_image)
    query_img = query_img.convert('RGB')
    query_img = query_img.resize((224, 224), Image.ANTIALIAS)

    input_img = V(centre_crop(query_img).unsqueeze(0), volatile=True)

    layer1_feature, layer2_feature, layer3_feature, layer4_feature = layer_feature(input_img, model)
    #print('shape : ',np.shape(layer1_feature),np.shape(layer2_feature),np.shape(layer3_feature),np.shape(layer4_feature))
    layer_feature_list = []

    layer_feature_list.append(layer1_feature.data.numpy())
    layer_feature_list.append(layer2_feature.data.numpy())
    layer_feature_list.append(layer3_feature.data.numpy())
    layer_feature_list.append(layer4_feature.data.numpy())

    query_features = []
    for feature_list in layer_feature_list:
        for i in range(len(feature_list[0])):
            a = feature_list[0][i].flatten()
            query_features.append(max(a))
    #print(np.shape(query_features))
    return query_features


def compared_feature(images, model):
    centre_crop = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(images)
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)

    input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

    layer1_feature, layer2_feature, layer3_feature, layer4_feature = layer_feature(input_img, model)
    #print('shape : ',np.shape(layer1_feature),np.shape(layer2_feature),np.shape(layer3_feature),np.shape(layer4_feature))
    layer_feature_list = []

    layer_feature_list.append(layer1_feature.data.numpy())
    layer_feature_list.append(layer2_feature.data.numpy())
    layer_feature_list.append(layer3_feature.data.numpy())
    layer_feature_list.append(layer4_feature.data.numpy())

    max_feature = []
    for feature_list in layer_feature_list:
        for i in range(len(feature_list[0])):
            a = feature_list[0][i].flatten()
            max_feature.append(max(a))
    #print(np.shape(max_feature))

    return max_feature


def detect(model_file):

    query_path = 'image/oxford_query'
    image_path = 'image/oxford'

    model = torch.load(model_file)

    model.eval()

    query_image_list = []
    result_name = []

    query_list = os.listdir(query_path)
    for query in query_list:
        #print(query)
        query_file = open(os.path.join(query_path, query), 'r')
        result_name.append(query)
        #print(result_name)
        for text in query_file:
            query_image = text.split(' ')[0] + '.jpg'
            query_image_list.append(query_image)
            #print(query_image_list)


    for c, query_image_ in enumerate(query_image_list):
        #print(query_image_)
        query_image = os.path.join(image_path, query_image_)
        #print(query_image)
        query_features = query_feature(query_image, model)
        similar_list = []
        save_result = open(result_name[c], 'w')
        print(result_name[c])
        image_list = os.listdir(image_path)
        image_list = np.sort(image_list)
        count = 0
        for images in image_list:
            #print(images)
            count += 1
            image = os.path.join(image_path, images)
            compared_features = compared_feature(image, model)
            similar_list.append([images, round(1 - spatial.distance.cosine(query_features, compared_features), 5)])
            if count % 1000 == 0:
                print(count)
        result = get_top_k_result(similar_list, 5063)
        #print(result)
        for i in result:
            save_result.write(str(i[0]))
            save_result.write('\n')

    return 0


if __name__ == '__main__':
    model_file = 'models/places365.pth.tar'

    result = detect(model_file)
    # print(result)