import os.path
import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import pandas as pd
import ntpath
import math
import glob
from PIL import Image
import cv2


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res


def process_pos_annotations_patient(src_path, patient_id):
    count = 0
    global count_1, count_2, count_3, count_4, count_5
    df_node = pandas.read_csv(settings.LUNA_ANOTATION)  # luna16的结节位置、尺寸大小
    nodule_path = settings.LUNA16_EXTRACTED_32_2DROI_DIR
    if not os.path.exists(nodule_path):
        os.mkdir(nodule_path)
    itk_img = SimpleITK.ReadImage(src_path)
    # src_path = './data/luna/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824.mhd'
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    img_array[img_array==-2048]=-1024
    # print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    # print("Annos in luna16 : ", len(df_patient))
    if not len(df_patient) == 0:
        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
        flip_direction_x = False
        flip_direction_y = False
        if round(direction[0]) == -1:
            origin[0] *= -1
            direction[0] = 1
            flip_direction_x = True
            # print("Swappint x origin")
        if round(direction[4]) == -1:
            origin[1] *= -1
            direction[4] = 1
            flip_direction_y = True
            # print("Swappint y origin")
        assert abs(sum(direction) - 3) < 0.01

        patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

        df_patient = df_node[df_node["seriesuid"] == patient_id]

        for index, annotation in df_patient.iterrows():
            node_x = annotation["coordX"]
            if flip_direction_x:
                node_x *= -1
            node_y = annotation["coordY"]
            if flip_direction_y:
                node_y *= -1
            node_z = annotation["coordZ"]
            diam_mm = annotation["diameter_mm"]  # 获取luna16的annotation的结节坐标、直径
            center_float = numpy.array([node_x, node_y, node_z])
            center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
            center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
            diameter_pixels = diam_mm / settings.TARGET_VOXEL_MM
            diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

            x1 = center_float_percent[0]  # luna16中的结节位置
            y1 = center_float_percent[1]
            z1 = center_float_percent[2]
            d1 = diameter_percent
            character = []

            cube = get_cube_from_img(patient_imgs,center_float_rescaled[0],center_float_rescaled[1],center_float_rescaled[2],32)
            nodule_extended = pandas.read_csv(settings.LUNA16_EXTRACTED_INFO_DIR + patient_id+"_annos_pos_lidc.csv")

            for index ,row in nodule_extended.iterrows():  # posline9中的结节位置与特征
                # print(f'id:{patient_id}', len(df_patient))
                x2=row["coord_x"]
                y2=row["coord_y"]
                z2=row["coord_z"]
                d2=row["diameter"]
                # print(x1, x2, y1, y2, z1, z2)

                dist=math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < (d1+d2)/2:
                    new_row=row[6:15].tolist()
                    new_row.append(round(annotation["diameter_mm"],2))
                    character.append(new_row)
                    count += 1
            character=numpy.array(character)
            # print(character)
            if not len(character)==0:
                # 对四位医生的注释求平均值
                character=numpy.round(character.mean(axis=0))
                # print(f'nodule:{patient_id},mlaignancy:{character[0]}')
                # print("nudule malignacne",character[0])

                if character[0]==5:
                    count_5 += 1
                    file_path = nodule_path+'/nodule_5'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    img = Image.fromarray(cube[16])  # 结节切片的层
                    img.save(file_path + '/' + str(patient_id) + '_' + str(count) +'.png')
                elif character[0]==4:
                    count_4 += 1
                    file_path = nodule_path+'/nodule_4'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    img = Image.fromarray(cube[16])  # 结节切片的层
                    img.save(file_path + '/' + str(patient_id) + '_' + str(count) +'.png')
                elif character[0]==3:
                    count_3 += 1
                    file_path = nodule_path+'/nodule_3'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    img = Image.fromarray(cube[16])  # 结节切片的层
                    img.save(file_path + '/' + str(patient_id) + '_' + str(count) +'.png')
                elif character[0]==2:
                    count_2 += 1
                    file_path = nodule_path+'/nodule_2'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    img = Image.fromarray(cube[16])  # 结节切片的层
                    img.save(file_path + '/' + str(patient_id) + '_' + str(count) +'.png')
                elif character[0]==1:
                    count_1 += 1
                    file_path = nodule_path+'/nodule_1'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    img = Image.fromarray(cube[16])  # 结节切片的层
                    img.save(file_path + '/' + str(patient_id) + '_' + str(count) +'.png')
                else:
                    print('ERROR || not 1-5', patient_id)
            else:
                print(f'id:{patient_id}',len(character))
                file_path = nodule_path + '/unkown'
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
                img = Image.fromarray(cube[16])  # 结节切片的层
                img.save(file_path + '/' + str(patient_id) + '_' + str(count) +'.png')
    else:
        # print("patiant",patient_id,"No nodules in annatations.csv")
        return None


# 在预处理代码中添加以下内容（保持原有流程不变）：
def generate_yolo_annotations(patient_id, annotations):
    """
    生成YOLO格式的标注文件
    参数：
        patient_id: 患者ID
        annotations: 包含结节坐标和尺寸的DataFrame
    返回：
        YOLO格式的标注字符串（每个结节一行）
    """
    yolo_annots = []
    for _, row in annotations.iterrows():
        # 转换为相对坐标和尺寸
        x_center = (row['coord_x'] + row['diameter'] / 2) / row['slice_width']
        y_center = (row['coord_y'] + row['diameter'] / 2) / row['slice_height']
        width = row['diameter'] / row['slice_width']
        height = row['coord_z'] / row['slice_depth']  # 处理深度维度

        # YOLO格式：class x_center y_center width height
        yolo_annots.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return '\n'.join(yolo_annots)


def save_ct_slice_and_annotation(img_slice, slice_info, patient_id, slice_idx):
    """
    保存CT切片和YOLO标注
    参数：
        img_slice: CT切片图像数据
        slice_info: 包含结节位置信息的字典
    """
    # 保存CT切片
    slice_path = os.path.join(settings.LUNA16_CT_SLICES_DIR,
                              f"{patient_id}_{slice_idx}.png")
    cv2.imwrite(slice_path, img_slice)

    # 生成并保存YOLO标注
    annot_path = os.path.join(settings.LUNA16_YOLO_LABELS_DIR,
                              f"{patient_id}_{slice_idx}.txt")
    with open(annot_path, 'w') as f:
        f.write(generate_yolo_annotations(patient_id, slice_info['nodules']))


if __name__ == "__main__":
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    only_patient = None
    candidate_index=0
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR+"subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            # src_path = './data/luna/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824.mhd'
            if only_patient is not None and only_patient not in src_path:
                continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            # print(candidate_index, " patient: ", patient_id)
            process_pos_annotations_patient(src_path, patient_id)
            candidate_index += 1
    print(count_1, count_2, count_3, count_4, count_5)