import os
from os import getcwd 
classes = ["disgust", "fear", "happiness", "others", "repression", "sadness", "surprise"]  
sets    = ["train", "test"]  

if __name__ == "__main__":  
    wd = getcwd()     
    print(wd)
    for se in sets:  
        list_file = open('cls_' + se + '.txt', 'w') 

        datasets_path = "datasets/" + se  #datasets/train
        types_name = os.listdir(datasets_path)  
        print(types_name)
        for type_name in types_name: #再一次循环
            if type_name not in classes:  # 判断语句
                continue
            cls_id = classes.index(type_name)  # classes.index按顺序输出
            print(cls_id)
            photos_path = os.path.join(datasets_path, type_name) 
            photos_name = os.listdir(photos_path)  
            # print(photos_name)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name) 
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                list_file.write(str(cls_id) + ";" + '%s/%s'%(wd, os.path.join(photos_path, photo_name)))  
                list_file.write('\n')
        list_file.close()

