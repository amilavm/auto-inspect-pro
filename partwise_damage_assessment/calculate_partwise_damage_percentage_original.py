# algorithm for the partwise damage percentage calculation

import cv2
import numpy as np

import os
from pathlib import Path

from shapely import geometry
from shapely.strtree import STRtree

def get_dmg_contours(directory, claim_folder_id):

    dmg_polys = []
    dmg_uid = []
    dmg_contour_list = []

    for im_file in os.listdir(directory+'/'+ claim_folder_id + '/damages'):
        if im_file.endswith(".png"):
            # print('check with file:', im_file)
            # read images - part
            img_dmg = cv2.imread(directory+'/'+claim_folder_id+'/damages/'+im_file)

            # convert to grayscale
            gray_dmg = cv2.cvtColor(img_dmg,cv2.COLOR_BGR2GRAY)

            # threshold and invert so polygon is white on black background
            thresh_dmg = cv2.threshold(gray_dmg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            thresh_dmg = 255 - thresh_dmg

            # get contours
            # result_part = np.zeros_like(img_dmg)
            contours_dmg = cv2.findContours(thresh_dmg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_dmg = contours_dmg[0] if len(contours_dmg) == 2 else contours_dmg[1]
            cntr_dmg = contours_dmg[0]
            aqueezed_polygon = geometry.Polygon(np.squeeze(cntr_dmg))
            dmg_polys.append(aqueezed_polygon)
            dmg_uid.append(Path(im_file).stem)
            dmg_contour_list.append(cntr_dmg)
            # final_list = list(zip(dmg_id, dmg_polys, cntr_part))
    return dmg_polys, dmg_uid, dmg_contour_list
    
def get_partwise_damage_report(directory, claim_folder_id):
    part_damage_list = []
    points, dmg_idntifier, contours_all = get_dmg_contours(directory, claim_folder_id)

    # category 1 - parts that may be replaced due to any type of damage / any severity level
    vehicle_parts_cat_1 = ['left_front_lamp', 'left_mirror', 'rear_screen', 'right_front_lamp',
    'right_mirror', 'right_tail_lights', 'windscreen']

    # category 2 - other parts that depend with damage level
    vehicle_parts_cat_2 = ['bonnet', 'front_bumper', 'grill', 'left_end_front_bumper', 
    'left_end_rear_bumper', 'left_fender', 'left_front_door', 'left_quarter_panel', 'left_rear_door', 'left_tail_lights', 
    'rear_bumper', 'rear_door', 'rear_spoiler', 'right_end_front_bumper', 
    'right_end_rear_bumper', 'right_fender', 'right_front_door', 
    'right_quarter_panel', 'right_rear_door']

    for im_file in os.listdir(directory+'/'+ claim_folder_id + '/parts'):
        if im_file.endswith(".png"):
            result = {}
            # read images - damage
            img_part = cv2.imread(directory+'/'+claim_folder_id+'/parts/'+im_file)

            # convert to grayscale
            gray_part = cv2.cvtColor(img_part,cv2.COLOR_BGR2GRAY)

            # threshold and invert so polygon is white on black background
            thresh_part = cv2.threshold(gray_part, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            thresh_part = 255 - thresh_part

            # get contours
            result_img = np.zeros_like(img_part)
            contours_part = cv2.findContours(thresh_part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_part = contours_part[0] if len(contours_part) == 2 else contours_part[1]
            cntr_part = contours_part[0]

            query_geom = geometry.Polygon(np.squeeze(cntr_part))

            # points, dmg_idntifier, contours_all = get_dmg_contours(directory, claim_folder_id)
            tree = STRtree(points)

            index_by_id = dict((id(pt), i) for i, pt in enumerate(points))
            lst = [(dmg_idntifier[index_by_id[id(pt)]], pt.wkt, contours_all[index_by_id[id(pt)]], pt.intersection(query_geom).area) for pt in tree.query(query_geom) if pt.intersects(query_geom)]
            # ('damage_type', 'pt', 'contour', 'intersected_area')
            # lst = [(name, pt.wkt) for name, pt in tree.query(query_geom) if pt.intersects(query_geom)]
            print(len(lst))
            # print(lst)
            # print(dmg_idntifier)
            if lst:
                dmg_found = [item[0] for item in lst]
                print(dmg_found)

                dmg_contour_found = [item[2] for item in lst]
                intersect_area = [item[3] for item in lst]

                print(intersect_area)

                part_area = query_geom.area
                intersection_area_sum = sum(intersect_area)
                ratio = intersection_area_sum/part_area * 100
                print('')
                print('part area: ', part_area)
                print('damage area: ', intersection_area_sum)
                print('damage ratio to the part: {:.2f}%'.format(ratio))

                cv2.drawContours(result_img, [cntr_part], 0, (255,255,255), 1)
                for i in dmg_contour_found:
                    cv2.drawContours(result_img, [i], 0, (255,255,255), 1)

                filename = Path(im_file).stem # same as the part type (filename=part)
                cv2.imwrite(directory+'/'+ claim_folder_id + '/processed_images' + '/' + filename + '_result_output.png', result_img) 


                #### Service Type ####
                if filename in vehicle_parts_cat_1:
                    if any("glass" in s for s in dmg_found):
                        service_type = 'replace'
                    elif any("broken" in s for s in dmg_found):
                        service_type = 'replace'
                    else:
                        service_type = 'repair'

                if filename in vehicle_parts_cat_2:
                    if any("dent" in s for s in dmg_found):
                        if (intersection_area_sum/part_area) > 0.5:
                            service_type = 'replace'
                        else:
                            service_type = 'repair'
                    elif any("broken" in s for s in dmg_found):
                        if (intersection_area_sum/part_area) < 0.15:
                            service_type = 'repair'
                        else:
                            service_type = 'replace'
                    else:
                        service_type = 'repair'

                result['part_name'] =  filename
                result['damage_percentage'] =  ratio
                result['service_type'] =  service_type
                part_damage_list.append(result)
    # print(part_damage_list)
    return part_damage_list

def get_final_assessment(files, claim_folder):
    final_result_list = []
    for file in files:
        filename = Path(file).stem
        output_list = get_partwise_damage_report(claim_folder, filename)
        final_result_list = final_result_list + output_list
    
    return final_result_list


# get_partwise_damage_report('archive', 'claim004')

