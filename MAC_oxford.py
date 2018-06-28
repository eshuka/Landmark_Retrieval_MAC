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


def query_feature(query_image, model):
    centre_crop = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #print(model)
    query_img = Image.open(query_image)
    query_img = query_img.convert('RGB')
    query_img = query_img.resize((224, 224), Image.ANTIALIAS)

    input_img = V(centre_crop(query_img).unsqueeze(0), volatile=True)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        input_img = input_img.cuda()

    feature_map = list(model.module.children())
    #print(feature_map)
    feature_map.pop()
    feature_map.pop()
    extractor = nn.Sequential(*feature_map)

    if use_gpu:
        model.cuda()
        extractor.cuda()
        cudnn.benchmark = True

    feature = extractor(input_img)

    query_features = []
    feature = feature.data.cpu().numpy()

    for i in range(len(feature[0])):
        a = feature[0][i].flatten()
        query_features.append(max(a))

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

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        input_img = input_img.cuda()

    feature_map = list(model.module.children())
    #print(feature_map)
    feature_map.pop()
    feature_map.pop()
    extractor = nn.Sequential(*feature_map)

    if use_gpu:
        model.cuda()
        extractor.cuda()
        cudnn.benchmark = True

    feature = extractor(input_img)

    max_feature = []
    feature = feature.data.cpu().numpy()
    # print(np.shape(feature))
    for i in range(len(feature[0])):
        a = feature[0][i].flatten()
        max_feature.append(max(a))

    return max_feature


def detect(model_file):
    # query_image = 'image/paris/defense/paris_defense_000605.jpg'
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
    model_file = 'models/paris.pth.tar'

    result = detect(model_file)
    # print(result)