from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import argparse
import os

from keras.applications import vgg19
from keras import backend as K
from tqdm import tqdm
import tensorflow as tf

conf = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 28, 'GPU': 1})
conf.gpu_options.allow_growth = True
conf.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=conf)
graph = tf.get_default_graph()
K.set_session(sess)

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str, default='./baseImages/img.jpg',
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str, default='./styleImages/img.jpeg',
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# pastrez aspect ratio, imaginea trebuie sa fie mai mica pentru a incapea in memorie
# 600 / width * img_nrows / height incape pe 2080Ti
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

from PIL import Image
import random


def get_white_noise_image(width, height):
    pil_map = Image.new("L", (width, height), 255)
    random_grid = map(lambda x: (random.randint(0, 255)), [0] * width * height)
    pil_map.putdata(list(random_grid))
    return pil_map.convert('RGB')


def white_noise_img():
    '''
    Functie folosita pentru generare de imagini din white noise.
    Am experimentat cu aceasta idee, nu a functionat foarte bine. Ideea este sa dam pondere foarte mare continutului,
    in asa fel incat atunci cand se genereaza imaginea sa inceapa sa apara formele, iar stilul implicit va fi aplicat
    odata cu formele. Imaginile sunt mult prea colorate, iar stilul nu se transfera bine.

    TODO: Sau este problema la parametrii?
    :return:
    '''
    img = get_white_noise_image(img_ncols, img_nrows)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def preprocess_image(image_path):
    '''
    Transforma o imagine intr un tensor de forma acceptata de VGG19
    :param image_path:
    :return:
    '''
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    '''
    Transforma un tensor intr-o imagine.
    :param x: tensorul
    :return: imaginea
    '''
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))


combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)


model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    '''
    Obtinem loss-ul dintre blocurile convolutionale si imaginile combinate, folosing matricile gram.

    :param style: imaginea din bloc
    :param combination: imaginile combinate
    :return:
    '''
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    '''

    :param base: Imaginea de input.
    :param combination: Tensorul de imagini combinate
    :return: MSE intre cele doua.
    '''
    return K.sum(K.square(combination - base))


def total_variation_loss(x):
    '''

    :param x: Tensorul imaginilor combinate.
    :return:
    '''
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
# adaugam loss-ul de continut pe e avand in vedere ponderea sa
loss = loss + content_weight * content_loss(base_image_features,
                                            combination_features)

# Vor prelua stilul din ultimele 5 blocuri convolutionale si vom calcula loss-ul pentru fiecare
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss = loss + (style_weight / len(feature_layers)) * sl     # adaugam loss-ul de stil, avand in vedere ponderea sa
loss = loss + total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

# x = white_noise_img()
x = preprocess_image(base_image_path)
pbar = tqdm(range(iterations))
prev = 0
mf = 10
mfr = mf*0.1
delta = 0
stwr = style_weight * 0.1
cr = content_weight * 0.1
fld = f'./Done/{result_prefix} {content_weight} {style_weight} {total_variation_weight}/'

if not os.path.isdir(fld):
    os.mkdir(fld)
for i in pbar:
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=mf)
    if i == 1:
        delta = (prev - min_val) * 0.2
    if i != 1 and prev - min_val < delta and prev != 0 and prev - min_val != 0:
        total_variation_weight -= total_variation_weight * 0.1
        # if style_weight > content_weight:
        #     style_weight -= stwr
        #     content_weight = 1-style_weight
        #     stwr = stwr * 0.1
        #
        # else:
        #     content_weight -= cr
        #     style_weight = 1-content_weight
        #     cr = cr * 0.1
        delta -= delta * 0.5
        mf += mfr
    prev = min_val
    pbar.set_description(desc=f"loss: {min_val} | mf={mf} | d={delta} | tvl={total_variation_weight} | stw={style_weight}")
    img = deprocess_image(x.copy())
    fname = fld + result_prefix + '_iteratia_%d.png' % i
    save_img(fname, img)