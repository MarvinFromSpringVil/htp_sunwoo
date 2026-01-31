import json 
import cv2
import os 
import numpy as np
import random 
import yaml
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

labels = [
    '집전체', '지붕', '문', '창문', '굴뚝', '연기',
    '사람전체', '머리', '상체', '다리', '팔', 
    '나무전체', '기둥', '수관', '뿌리', '열매'
]

kor2eng = {
    '집전체':'House',
    '지붕':'Roof',
    '집벽':'Wall',
    '문':'Door',
    '창문':'Window',
    '굴뚝':'Chimney',
    '연기':'ChimneySmoke',

    '사람전체': 'Person',
    '머리': 'Head',
    '얼굴': 'Face',
    '상체': 'Body',
    '다리': 'Leg',
    '팔': 'Arm',

    '나무전체': 'Tree',
    '기둥': 'Pole',
    '수관': 'Crown',
    '뿌리': 'Root',
    '나뭇잎': 'Leaf',
    '꽃': 'Flower',
    '열매': 'Fruit'
}

def _parse_size(str_size):
    xy = str_size.split('x')
    return int(xy[0]) * int(xy[1])

class StatAnalysis:
    def __init__(self, stat_data_path) -> None:
        self.stat_data_house = load_json(stat_data_path['house'])
        self.stat_data_tree = load_json(stat_data_path['tree'])
        self.stat_data_person = load_json(stat_data_path['person'])

        # cfg
        with open("config.yaml", "r") as file:
            data = yaml.safe_load(file)

        for gender in ['total', 'female', 'male']:
            for k in self.stat_data_house[gender]:
                if not k in data['house']:
                    del self.stat_data_house[gender][k]

            for k in self.stat_data_tree[gender]:
                if not k in data['tree']:
                    del self.stat_data_tree[gender][k]

            for k in self.stat_data_person[gender]:
                if not k in data['person']:
                    del self.stat_data_person[gender][k]
            
    def drawGraphs(self, age, gender, inference_result, processed_images):
        self.age = age 
        self.gender = gender

        self.ret = {}

        # house
        for k in inference_result:
            if k in self.stat_data_house[gender]:
                file_name = self._draw_dist(k, self.stat_data_house[gender][k], inference_result[k])
                processed_images.append(file_name)

                mu, percentile = self._compute_percentile(self.stat_data_house[gender][k], inference_result[k])
                self.ret[k] = (mu, percentile)

        # person 
        for k in inference_result:
            if k in self.stat_data_person[gender]:
                file_name = self._draw_dist(k, self.stat_data_person[gender][k], inference_result[k])
                processed_images.append(file_name)

                mu, percentile = self._compute_percentile(self.stat_data_person[gender][k], inference_result[k])
                self.ret[k] = (mu, percentile)

        # tree
        for k in inference_result:
            if k in self.stat_data_tree[gender]:
                file_name = self._draw_dist(k, self.stat_data_tree[gender][k], inference_result[k])
                processed_images.append(file_name)

                mu, percentile = self._compute_percentile(self.stat_data_tree[gender][k], inference_result[k])
                self.ret[k] = (mu, percentile)

    def get_text_result(self, age, gender, inference_result):
        text = '{} ({}) HTP test result.'.format(age, gender)
        '''
        for k in self.ret:
            mu, percentile = self.ret[k]
            text += '{}: avg {}, %: {:.1f}'.format(k, mu, percentile)
        '''
        return text

    def _compute_percentile(self, total_stat, sample_value):
        mean_value = np.mean(total_stat) 

        percentile = 100 * (1 - np.sum(np.array(total_stat) <= sample_value) / len(total_stat))
        return mean_value, percentile

    def _draw_dist(self, title, total_stat, sample_value, bins=30, color='skyblue'):
        # Create a histogram
        n, bins, patches = plt.hist(total_stat, bins=bins, edgecolor='black', color='gray')    
        bin_index = np.digitize(sample_value, bins) - 1  # np.digitize는 1-based index 반환

        patches[bin_index].set_facecolor('black')

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(title)
        
        file_name = 'dist_{}.jpg'.format(title)

        dst_path = os.path.join('./static/processed/dist_{}.jpg'.format(title))
        plt.savefig(dst_path, dpi=300, bbox_inches='tight')
        plt.close() 

        return file_name

def _draw_text_above_bbox(image, bbox, text_data, font_scale=0.5, color=(255, 0, 0), thickness=1):
    x, y, w, h = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size to adjust background if necessary
    (text_width, text_height), baseline = cv2.getTextSize(text_data, font, font_scale, thickness)

    # Text position: slightly above the bounding box
    text_x = x
    text_y = max(y - 5, text_height + 5)  # Ensure text is within image bounds

    # Optionally, you can draw a filled rectangle for better visibility
    # Uncomment the next line if you want a background for the text
    # cv2.rectangle(image, (text_x, text_y - text_height), (text_x + text_width, text_y + baseline), (0, 0, 0), -1)

    # Put the text
    cv2.putText(image, text_data, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def load_json(json_path):
    data = None 
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data 

def analysis_bbox(img_path):
    img = cv2.imread(img_path)
    json_path = img_path.replace('jpg', 'json')

    ret = {}

    data = load_json(json_path)
    image_size = _parse_size(data['meta']['img_resolution'])

    person_size = None 
    person_height = None 
    person_width = None
    tree_size = None 
    #house_size = None 

    for bbox in data['annotations']['bbox']:
        label = bbox['label']
        w = bbox['w']
        h = bbox['h']

        if label == '나무전체': 
            tree_size = w*h 
            break 
        elif label == '사람전체':
            person_size = w*h
            person_height = h 
            person_width = w 
            break 


    for bbox in data['annotations']['bbox']:
        label = bbox['label']
        if not label in labels:
            continue
        w = bbox['w']
        h = bbox['h']
        
        # house 
        if label == '집전체':
            ret['house_prop'] = w*h / image_size
        elif label == '연기':
            ret['chimney_smoke'] = True
        elif label == '지붕':
            ret['roof_prop'] = w*h / image_size
        elif label == '문':
            ret['door'] = True
            ret['door_prop'] = w*h / image_size
        elif label == '창문':
            if 'window' in ret:
                ret['window'] += 1
            else:
                ret['window'] = 1
            ret['window_prop'] = w*h / image_size
        
        # tree
        elif label == '기둥':
            ret['pole_prop'] = w*h / tree_size
        elif label == '수관':
            ret['root_prop'] = w*h / tree_size
        elif label == '뿌리':
            ret['crown_prop'] = w*h / tree_size
        elif label == '나뭇잎':
            if 'leaf' in ret:
                ret['leaf'] += 1
            else:
                ret['leaf'] = 1
        elif label == '꽃':
            if 'flower' in ret:
                ret['flower'] += 1
            else:
                ret['flower'] = 1
        elif label == '열매':
            if 'fruit' in ret:
                ret['fruit'] += 1
            else:
                ret['fruit'] = 1
        
        # person 
        elif label == '머리':
            ret['head_prop'] = w*h / person_size
        elif label == '얼굴':
            ret['face_prop'] = w*h / person_size
        elif label == '상체':
            ret['body_prop'] = w*h / person_size
        elif label == '다리':
            ret['leg_height_prop'] = h / person_height
        elif label == '팔':
            ret['arm_width_prop'] = w / person_width

    return ret
       

def draw_bbox(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None 
    
    json_path = img_path.replace('jpg', 'json')

    data = load_json(json_path)
    if data is None:
        return None 
    
    for bbox in data['annotations']['bbox']:
        label = bbox['label']
        if not label in labels:
            continue

        x = bbox['x']
        y = bbox['y']
        w = bbox['w']
        h = bbox['h']
        
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        cv2.rectangle(img, (x,y), (x+w, y+h), (b,g,r), 3)
        _draw_text_above_bbox(img, (x, y, w, h), kor2eng[label], font_scale=2, color=(0, 0, 255), thickness=2)

    return img


