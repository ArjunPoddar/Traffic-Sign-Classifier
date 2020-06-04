import numpy as np
import cv2

def zoom_in_image(image):
    '''
    input: an image
    
    output: scaled version of the image
    '''
    try:
        in_h, in_w = image.shape[1], image.shape[0]
        x = np.random.normal(0.03, 0.01, 1)
        delta_h = int(np.round(x*in_h))
        y = np.random.normal(0.03, 0.01, 1)
        delta_w = int(np.round(y*in_w))
        out_image = image[delta_h: in_h - delta_h, delta_w: in_w - delta_w]
        out_image = cv2.resize(out_image, dsize = (in_h, in_w))
        return out_image
    except:
        return image

def zoom_out_image(image):
    '''
    input: an image
    
    output: scaled version of the image
    '''
    try:
        in_h, in_w = image.shape[1], image.shape[0]
        x = np.random.normal(0.03, 0.01, 1)
        delta_h = int(np.round(x*in_h))
        y = np.random.normal(0.03, 0.01, 1)
        delta_w = int(np.round(y*in_w))
        out_image = np.copy(image)
        out_image = cv2.resize(out_image, dsize = (in_h*(1-x), in_w*(1-y)))
        top = (in_h - out_image.shape[1])// 2
        bottom = in_h - (out_image.shape[1] + top)
        left =  (in_w - out_image.shape[0])// 2
        right = in_w - (out_image.shape[0] + left)
        out_image = cv2.copyMakeBorder(out_image, left, right, top, bottom, cv2.BORDER_REFLECT)
        return out_image
    except:
        return image

def zoom_in_or_out(image):
    zoom = np.random.choice(['in', 'out'], 1, p = [0.5, 0.5])
    try:
        if zoom == 'in':
            out_image = zoom_in_image(image)
        else:
            out_image = zoom_out_image(image)
    except:
        out_image = image
    return out_image

def translate_image(image):
    '''
    input: an image
    
    output: randomly translated image in x and y directions
    '''
    in_h, in_w = image.shape[1], image.shape[0]
    pixels_allowed = 3
    delta_x, delta_y = np.random.randint(-pixels_allowed, pixels_allowed, 2)
    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    out_image = cv2.warpAffine(image, M, (in_w, in_h))
    return out_image

def rotate_image(image):
    '''
    input: an image
    
    output: randomly translated image in x and y directions
    source: https://stackoverflow.com/a/9042907
    '''
    angle_allowed = 30
    angle = np.random.randint(-angle_allowed, angle_allowed, 1)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    out_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return out_image