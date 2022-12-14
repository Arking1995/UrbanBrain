from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
import numpy as np
import shapely.affinity as sa
import matplotlib.pyplot as plt

blk_color = [0.9490196078431372, 0.9372549019607843, 0.9137254901960784]
road_contour_color = np.array(blk_color) * 0.6
bldg_colors = [(0.6509803921568628, 0.807843137254902, 0.8901960784313725), (0.12156862745098039, 0.47058823529411764, 0.7058823529411765), (0.6980392156862745, 0.8745098039215686, 0.5411764705882353), (0.2, 0.6274509803921569, 0.17254901960784313), (0.984313725490196, 0.6039215686274509, 0.6), (0.8901960784313725, 0.10196078431372549, 0.10980392156862745), (0.9921568627450981, 0.7490196078431373, 0.43529411764705883), (1.0, 0.4980392156862745, 0.0), (0.792156862745098, 0.6980392156862745, 0.8392156862745098), (0.41568627450980394, 0.23921568627450981, 0.6039215686274509), (1.0, 1.0, 0.6), (0.6941176470588235, 0.34901960784313724, 0.1568627450980392)]
bldg_color = [0.85098039, 0.81568627, 0.78823529]
bldg_contour_color = np.array(bldg_color) * 0.6

def _azimuth(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""
    import numpy as np

    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

def _dist(a, b):
    """distance between points"""
    import math

    return math.hypot(b[0] - a[0], b[1] - a[1])

def get_azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    mrr = mrr.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])
    return az


def get_block_parameters(block):
    bbx = block.minimum_rotated_rectangle
    azimuth = get_azimuth(bbx)
    return azimuth, bbx
    

def norm_block_to_horizonal(bldg, azimuth, bbx):
    blk_offset_x = np.double(bbx.centroid.x)
    blk_offset_y = np.double(bbx.centroid.y)

    for i in range(len(bldg)):
        curr = sa.translate(bldg[i], -blk_offset_x, -blk_offset_y)
        bldg[i] = sa.rotate(curr, azimuth - 90, origin = (0.0, 0.0))

    return bldg


########################### Input position and size from graph output, and block, and midaxis. Output original bldg with correct position/size/rotation ################################################################
def inverse_warp_bldg_by_midaxis(pos_sorted, size_sorted, midaxis, aspect_rto, rotate_bldg_by_midaxis = True, output_mode = False):
    org_size = np.zeros_like(pos_sorted)
    org_pos = np.zeros_like(pos_sorted)

    pos_sorted[:, 0] = (pos_sorted[:, 0] + 1.0) / 2.0 ############ normalize pos_x [-1, 1] back to [0, 1], pos_y keep [-1, 1] 
    pos_sorted[:, 1] = pos_sorted[:, 1] / 2.0
    size_sorted = size_sorted / 2.0

    midaxis_length = midaxis.length
    mean_block_width = aspect_rto * midaxis_length
    org_size[:, 0] = size_sorted[:, 0] * midaxis_length

    ###############################################################################   same as forward processing   ###################
    relative_cutoff = [0.0]
    vector_midaxis = []
    coords = np.array(midaxis.coords.xy)
    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])
    relative_cutoff = np.array(relative_cutoff)
    vector_midaxis = np.array(vector_midaxis)

    bldgnum = pos_sorted.shape[0]        
    corres_midaxis_vector_idx = []
    for i in range(bldgnum):
        cur_x = pos_sorted[i, 0]
        insert_pos = get_insert_position(relative_cutoff, cur_x) - 1
        if insert_pos > vector_midaxis.shape[0]-1:
            print('\n out of index in vector_midaxis. \n')
            corres_midaxis_vector_idx.append(vector_midaxis.shape[0]-1)
            continue
        corres_midaxis_vector_idx.append(insert_pos)
    corres_midaxis_vector_idx = np.array(corres_midaxis_vector_idx)
    ###############################################################################   same as forward processing   ###################

    ###############################################################################   get correct position and size   ###################
    for i in range(bldgnum):
        vertical_point_on_midaxis = midaxis.interpolate(pos_sorted[i, 0], normalized=True)
        cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
        if pos_sorted[i, 1] <= 0:
            vec_from_midaxis_to_bldg = np.array([cur_vector_midaxis[1], -cur_vector_midaxis[0]])
        else:
            vec_from_midaxis_to_bldg = np.array([-cur_vector_midaxis[1], cur_vector_midaxis[0]])
        
        vec_from_midaxis_to_bldg = vec_from_midaxis_to_bldg / np.linalg.norm(vec_from_midaxis_to_bldg)

        cur_pos_x = vertical_point_on_midaxis.x + vec_from_midaxis_to_bldg[0] * np.abs(pos_sorted[i, 1]) * (mean_block_width)
        cur_pos_y = vertical_point_on_midaxis.y + vec_from_midaxis_to_bldg[1] * np.abs(pos_sorted[i, 1]) * (mean_block_width)


        org_pos[i, 0], org_pos[i, 1] = cur_pos_x, cur_pos_y
        org_size[i, 1] = size_sorted[i, 1] * mean_block_width   ##   changed from multiply by "line_from_midaxis_to_contour.length"
    ###############################################################################   get correct position and size   ###################
    if output_mode:
        org_pos, org_size = modify_pos_size_arr_overlap(org_pos, org_size)

    ###############################################################################   get original rotation  ###################
    org_bldg = []
    for i in range(org_pos.shape[0]):
        curr_bldg = Polygon(get_bbx(org_pos[i,:], org_size[i,:]))
        if rotate_bldg_by_midaxis:
            cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
            angle = np.arctan2(cur_vector_midaxis[1], cur_vector_midaxis[0]) * 180.0 / np.pi
            curr_bldg = sa.rotate(curr_bldg, angle, origin=(org_pos[i,0], org_pos[i,1]))
        org_bldg.append(curr_bldg)
    
    return org_bldg , org_pos, org_size
#######################################################################################################################


###############################################################
def get_insert_position(arr, K):
    # Traverse the array
    for i in range(arr.shape[0]):         
        # If K is found
        if arr[i] == K:
            return np.int16(i)
        # If arr[i] exceeds K
        elif arr[i] >= K:
            return np.int16(i)
    return np.int16(arr.shape[0])



def get_bbx(pos, size):
    bl = ( pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0)
    br = ( pos[0] + size[0] / 2.0, pos[1] - size[1] / 2.0)
    ur = ( pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0)
    ul = ( pos[0] - size[0] / 2.0, pos[1] + size[1] / 2.0)
    return bl, br, ur, ul


def modify_pos_size_arr_overlap(pos, size, iou_threshold = 0.5, height = None):    # input pos and size np.array, the same as shapely geometry version "modify_geometry_overlap".
    bldgnum = pos.shape[0]
    pos = np.array(pos)
    size = np.array(size)
    rm_list = []
    bldg = []

    for i in range(pos.shape[0]):
        bldg.append(box(pos[i,0] - size[i,0] / 2.0, pos[i,1] - size[i,1] / 2.0, pos[i,0] + size[i,0] / 2.0, pos[i,1] + size[i,1] / 2.0))

    for i in range(bldgnum):
        for j in range(i+1, bldgnum):
            is_mod = False
            p1 = bldg[i]
            p2 = bldg[j]
            if p1.contains(p2):
                rm_list.append(i)
                continue
            if p2.contains(p1):
                rm_list.append(j)
                continue
            if p1.intersects(p2):
                intersect = p1.intersection(p2)
                int_area = intersect.area
                iou1 = int_area / (p1.area + 1e-6)
                iou2 = int_area / (p2.area + 1e-6)

                if iou1 > iou_threshold:
                    rm_list.append(i)
                    continue
                else:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)
                    is_mod = True

                if iou2 > iou_threshold:
                    rm_list.append(j)
                    continue
                elif not is_mod:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)

    for i in range(bldgnum):
       if np.fabs(size[i, 0]) < 1e-2 or np.fabs(size[i, 1]) < 1e-2:
            rm_list.append(i)
                
    pos = np.delete(pos, rm_list, axis=0)
    size = np.delete(size, rm_list, axis=0)

    if height is None:
        return pos, size
    else:
        height = np.delete(height, rm_list, axis=0)
        return pos, size, height


def remove_mutual_overlap(pos, size, intersect):
    int_bbx = intersect.bounds
    x_d = int_bbx[2] - int_bbx[0]
    y_d = int_bbx[3] - int_bbx[1]
    cx_d = intersect.centroid.x
    cy_d = intersect.centroid.y

    s_x = size[0]
    s_y = size[1]

    if np.double(x_d) / np.double(s_x) >= np.double(y_d) / np.double(s_y):
        if pos[1] >= cy_d:
            pos[1] = pos[1] + y_d / 2.0
        else:
            pos[1] = pos[1] - y_d / 2.0
        size[1] = size[1] - y_d
    else:
        if pos[0] >= cx_d:
            pos[0] = pos[0] + x_d / 2.0
        else:
            pos[0] = pos[0] - x_d / 2.0
        size[0] = size[0] - x_d

    return pos, size


###############################################################
def get_block_aspect_ratio(block, midaxis):
    coords = np.array(midaxis.coords.xy)
    midaxis_length = midaxis.length

    ################################################################
    relative_cutoff = [0.0]
    vector_midaxis = []
    block_width_list = []

    if midaxis.geom_type == 'GeometryCollection':
        for jj in list(midaxis.geoms):
            if jj.geom_type == 'LineString':
                midaxis = jj
                break
    coords = np.array(midaxis.coords.xy)

    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])

        if i < coords.shape[1] - 1:
            cur_width = get_block_width_from_pt_on_midaxis(block, coords[:, i] - coords[:, i-1], Point(coords[0, i], coords[1, i])) # each node on midaxis, except the last and the front.
            block_width_list.append(cur_width)

    if block_width_list == []:
        mean_block_width = block.bounds[3] - block.bounds[1]
    else:
        block_width_list = np.array(block_width_list)
        mean_block_width = np.mean(block_width_list)

    aspect_rto = np.double(mean_block_width) / np.double(midaxis_length)
    return aspect_rto


################################################################
def get_block_width_from_pt_on_midaxis(block, vector_midaxis, pt_on_midaxis):
    unit_v =  vector_midaxis / np.linalg.norm(vector_midaxis)
    left_v = np.array([-unit_v[1], unit_v[0]])
    right_v = np.array([unit_v[1], -unit_v[0]])
    
    dummy_left_pt = Point(pt_on_midaxis.x + left_v[0], pt_on_midaxis.y + left_v[1])
    dummy_right_pt = Point(pt_on_midaxis.x + right_v[0], pt_on_midaxis.y + right_v[1])

    left_line_to_contour = get_extend_line(dummy_left_pt, pt_on_midaxis, block, False, is_extend_from_end = True)
    right_line_to_contour = get_extend_line(dummy_right_pt, pt_on_midaxis, block, False, is_extend_from_end = True)

    # print(left_line_to_contour.length, right_line_to_contour.length)
    
    return left_line_to_contour.length + right_line_to_contour.length



def get_extend_line(a, b, block, isfront, is_extend_from_end = False):
    minx, miny, maxx, maxy = block.bounds
    if a.x == b.x:  # vertical line
        if a.y <= b.y:
            extended_line = LineString([a, (a.x, minx)])
        else:
            extended_line = LineString([a, (a.x, maxy)])
    elif a.y == b.y:  # horizonthal line
        if a.x <= b.x:
            extended_line = LineString([a, (minx, a.y)])
        else:
            extended_line = LineString([a, (maxx, a.y)])

    else:
        # linear equation: y = k*x + m
        k = (b.y - a.y) / (b.x - a.x)
        m = a.y - k * a.x
        if k >= 0:
            if b.x - a.x >= 0:
                y1 = k * minx + m
                x1 = (miny - m) / k
                points_on_boundary_lines = [Point(minx, y1), Point(x1, miny)]
            else:
                y1 = k * maxx + m
                x1 = (maxy - m) / k
                points_on_boundary_lines = [Point(maxx, y1), Point(x1, maxy)]        
        else:
            if b.x - a.x >= 0:
                y1 = k * minx + m
                x1 = (maxy - m) / k
                points_on_boundary_lines = [Point(minx, y1), Point(x1, maxy)]
            else:
                y1 = k * maxx + m
                x1 = (miny - m) / k
                points_on_boundary_lines = [Point(maxx, y1), Point(x1, miny)]

        # print('points on bound: ', points_on_boundary_lines[0].coords.xy, points_on_boundary_lines[1].coords.xy)
        
        points_sorted_by_distance = sorted(points_on_boundary_lines, key=a.distance)
        extended_line = LineString([a, Point(points_sorted_by_distance[0])])
    

    min_dis = 9999999999.9
    intersect = block.boundary.intersection(extended_line)
    if intersect.geom_type == 'MultiPoint':
        for i in intersect:
            if i.distance(a) <= min_dis:
                nearest_points_on_contour = i
    elif intersect.geom_type == 'Point':
        nearest_points_on_contour = intersect
    # elif intersect.geom_type == 'LineString':
    #     if not is_extend_from_end:
    #         nearest_points_on_contour = a
    #     else:
    #         nearest_points_on_contour = b
    else:
        if not is_extend_from_end:
            nearest_points_on_contour = a
        else:
            nearest_points_on_contour = b
        print('intersect: ', intersect)
        print('unknow geom type on intersection: ', intersect.geom_type)

    if not is_extend_from_end:
        if isfront:
            line_to_contour = LineString([nearest_points_on_contour, a])
        else:
            line_to_contour = LineString([a, nearest_points_on_contour])
    else:
        if isfront:
            line_to_contour = LineString([nearest_points_on_contour, b])
        else:
            line_to_contour = LineString([b, nearest_points_on_contour])

    return line_to_contour




def get_geometry_by_shape_iou(shape_type, iou, bldg):
    rot_rect = bldg.minimum_rotated_rectangle

    if shape_type == 0 or shape_type == 1 or shape_type == 5:  # rectangle
        scale_x = np.random.normal(np.sqrt(iou), 0.02, 1)
        scale_y = iou / scale_x
        out_bldg = sa.scale(rot_rect, xfact = scale_x, yfact = scale_y)
        if out_bldg.geom_type != 'Polygon':
            return bldg
    

    if shape_type == 2:   # cross shape
        offset_area_list = np.random.random(4)
        offset_area_list = (1.0-iou) * offset_area_list / np.sum(offset_area_list)
        offset_x_list = []
        offset_y_list = []

        for i in range(4):
            rd_asp_rto = np.random.normal(0.5, 0.1, 1)[0] + 1e-3 #np.random.random() + 1e-3
            offset_area = offset_area_list[i]
            offset_x1 = np.sqrt(offset_area / rd_asp_rto)
            offset_y1 = offset_x1 * rd_asp_rto
            offset_x_list.append(offset_x1)
            offset_y_list.append(offset_y1)
       
        out_bldg = get_cross_geometry(offset_x_list, offset_y_list, rot_rect)

    
    if shape_type == 3:  # L-shape
        offset_area = 1.0 - iou
        rd_asp_rto = np.random.random() + 1e-3
        offset_x = np.sqrt(offset_area / rd_asp_rto)
        offset_y = offset_x * rd_asp_rto
        out_bldg = get_L_geometry(offset_x, offset_y, rot_rect)
    


    if shape_type == 4:  # U-shape
        offset_area = 1.0 - iou
        rd_asp_rto = np.random.normal(0.5, 0.1, 1)[0]
        offset_x = np.sqrt(offset_area / rd_asp_rto)
        offset_y = offset_x * rd_asp_rto
        out_bldg = get_U_geometry(offset_x, offset_y, rot_rect)
    
    return out_bldg  



def get_cross_geometry(offset_x_list, offset_y_list, rot_rect):

    if offset_x_list[0] + offset_x_list[1] > 0.95:
        offset_x_list[1] = 0.95 - offset_x_list[0]
    if offset_x_list[2] + offset_x_list[3] > 0.95:
        offset_x_list[2] = 0.95 - offset_x_list[3]

    if offset_y_list[1] + offset_y_list[2] > 0.95:
        offset_y_list[2] = 0.95 - offset_y_list[1]

    if offset_y_list[3] + offset_y_list[0] > 0.95:
        offset_y_list[3] = 0.95 - offset_y_list[0]

    if rot_rect.geom_type != 'Polygon':
        return rot_rect

    o_pt1 = Point(rot_rect.exterior.coords[0])
    o_pt2 = Point(rot_rect.exterior.coords[1])
    o_pt3 = Point(rot_rect.exterior.coords[2])
    o_pt4 = Point(rot_rect.exterior.coords[3])

    line1 = LineString([o_pt1, o_pt2])  
    line2 = LineString([o_pt1, o_pt4])
    pt1 = line2.interpolate(offset_y_list[0], normalized = True)
    pt3 = line1.interpolate(offset_x_list[0], normalized = True)
    vec_01_3 = np.array([pt3.x - o_pt1.x, pt3.y - o_pt1.y])
    vec_01_1 = np.array([pt1.x - o_pt1.x, pt1.y - o_pt1.y])
    pt2 = Point([o_pt1.x + vec_01_1[0] + vec_01_3[0], o_pt1.y + vec_01_1[1] + vec_01_3[1]])


    line3 = LineString([o_pt2, o_pt1])  
    line4 = LineString([o_pt2, o_pt3])
    pt4 = line3.interpolate(offset_x_list[1], normalized = True)
    pt6 = line4.interpolate(offset_y_list[1], normalized = True)
    vec_02_4 = np.array([pt4.x - o_pt2.x, pt4.y - o_pt2.y])
    vec_02_6 = np.array([pt6.x - o_pt2.x, pt6.y - o_pt2.y])
    pt5 = Point([o_pt2.x + vec_02_4[0] + vec_02_6[0], o_pt2.y + vec_02_4[1] + vec_02_6[1]])


    line5 = LineString([o_pt3, o_pt2])  
    line6 = LineString([o_pt3, o_pt4])
    pt7 = line5.interpolate(offset_y_list[2], normalized = True)
    pt9 = line6.interpolate(offset_x_list[2], normalized = True)
    vec_03_7 = np.array([pt7.x - o_pt3.x, pt7.y - o_pt3.y])
    vec_03_9 = np.array([pt9.x - o_pt3.x, pt9.y - o_pt3.y])
    pt8 = Point([o_pt3.x + vec_03_7[0] + vec_03_9[0], o_pt3.y + vec_03_7[1] + vec_03_9[1]])


    line7 = LineString([o_pt4, o_pt3])  
    line8 = LineString([o_pt4, o_pt1])
    pt10 = line7.interpolate(offset_x_list[3], normalized = True)
    pt12 = line8.interpolate(offset_y_list[3], normalized = True)
    vec_04_10 = np.array([pt10.x - o_pt4.x, pt10.y - o_pt4.y])
    vec_04_12 = np.array([pt12.x - o_pt4.x, pt12.y - o_pt4.y])
    pt11 = Point([o_pt4.x + vec_04_10[0] + vec_04_12[0], o_pt4.y + vec_04_10[1] + vec_04_12[1]])

    output_cross = Polygon([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11, pt12])

    return output_cross



def get_L_geometry(offset_x, offset_y, rot_rect):
    rd_id = int(np.random.randint(0, high=4, size=1)[0])

    if rot_rect.geom_type != 'Polygon':
        return rot_rect

    o_pt1 = Point(rot_rect.exterior.coords[0])
    o_pt2 = Point(rot_rect.exterior.coords[1])
    o_pt3 = Point(rot_rect.exterior.coords[2])
    o_pt4 = Point(rot_rect.exterior.coords[3])

    line1 = LineString([o_pt1, o_pt2])  
    line2 = LineString([o_pt1, o_pt4])

    line3 = LineString([o_pt2, o_pt1])  
    line4 = LineString([o_pt2, o_pt3])

    line5 = LineString([o_pt3, o_pt2])  
    line6 = LineString([o_pt3, o_pt4])

    line7 = LineString([o_pt4, o_pt3])  
    line8 = LineString([o_pt4, o_pt1])

    if rd_id == 0:
        pt1 = line2.interpolate(offset_y, normalized = True)
        pt3 = line1.interpolate(offset_x, normalized = True)
        vec_01_3 = np.array([pt3.x - o_pt1.x, pt3.y - o_pt1.y])
        vec_01_1 = np.array([pt1.x - o_pt1.x, pt1.y - o_pt1.y])
        pt2 = Point([o_pt1.x + vec_01_1[0] + vec_01_3[0], o_pt1.y + vec_01_1[1] + vec_01_3[1]])
        output_L = Polygon([pt1, pt2, pt3, o_pt2, o_pt3, o_pt4])

    if rd_id == 1:
        line3 = LineString([o_pt2, o_pt1])  
        line4 = LineString([o_pt2, o_pt3])
        pt4 = line3.interpolate(offset_x, normalized = True)
        pt6 = line4.interpolate(offset_y, normalized = True)
        vec_02_4 = np.array([pt4.x - o_pt2.x, pt4.y - o_pt2.y])
        vec_02_6 = np.array([pt6.x - o_pt2.x, pt6.y - o_pt2.y])
        pt5 = Point([o_pt2.x + vec_02_4[0] + vec_02_6[0], o_pt2.y + vec_02_4[1] + vec_02_6[1]])
        output_L = Polygon([o_pt1, pt4, pt5, pt6, o_pt3, o_pt4])

    if rd_id == 2:
        line5 = LineString([o_pt3, o_pt2])  
        line6 = LineString([o_pt3, o_pt4])
        pt7 = line5.interpolate(offset_y, normalized = True)
        pt9 = line6.interpolate(offset_x, normalized = True)
        vec_03_7 = np.array([pt7.x - o_pt3.x, pt7.y - o_pt3.y])
        vec_03_9 = np.array([pt9.x - o_pt3.x, pt9.y - o_pt3.y])
        pt8 = Point([o_pt3.x + vec_03_7[0] + vec_03_9[0], o_pt3.y + vec_03_7[1] + vec_03_9[1]])
        output_L = Polygon([o_pt1, o_pt2, pt7, pt8, pt9, o_pt4])
    
    if rd_id == 3:
        line7 = LineString([o_pt4, o_pt3])  
        line8 = LineString([o_pt4, o_pt1])
        pt10 = line7.interpolate(offset_x, normalized = True)
        pt12 = line8.interpolate(offset_y, normalized = True)
        vec_04_10 = np.array([pt10.x - o_pt4.x, pt10.y - o_pt4.y])
        vec_04_12 = np.array([pt12.x - o_pt4.x, pt12.y - o_pt4.y])
        pt11 = Point([o_pt4.x + vec_04_10[0] + vec_04_12[0], o_pt4.y + vec_04_10[1] + vec_04_12[1]])
        output_L = Polygon([o_pt1, o_pt2, o_pt3, pt10, pt11, pt12])

    return output_L




def get_U_geometry(offset_x, offset_y, rot_rect):
    if rot_rect.geom_type != 'Polygon':
        return rot_rect
        
    rd_id = int(np.random.randint(0, high=4, size=1)[0])

    o_pt1 = Point(rot_rect.exterior.coords[0])
    o_pt2 = Point(rot_rect.exterior.coords[1])
    o_pt3 = Point(rot_rect.exterior.coords[2])
    o_pt4 = Point(rot_rect.exterior.coords[3])

    line1 = LineString([o_pt1, o_pt2])  
    line2 = LineString([o_pt1, o_pt4])

    line3 = LineString([o_pt2, o_pt1])  
    line4 = LineString([o_pt2, o_pt3])

    line5 = LineString([o_pt3, o_pt2])  
    line6 = LineString([o_pt3, o_pt4])

    line7 = LineString([o_pt4, o_pt3])  
    line8 = LineString([o_pt4, o_pt1])

    if rd_id == 0:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)
        pt1 = Point(line1.interpolate(interp, normalized=True))
        dummy_x = Point(line1.interpolate(offset_x, normalized=True))
        dummy_y = Point(line2.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt1.x, dummy_x.y - o_pt1.y])
        vec_y = np.array([dummy_y.x - o_pt1.x, dummy_y.y - o_pt1.y])
        pt2 = Point((pt1.x + vec_y[0], pt1.y + vec_y[1]))
        pt3 = Point((pt2.x + vec_x[0], pt2.y + vec_x[1]))
        pt4 = Point((pt3.x - vec_y[0], pt3.y - vec_y[1]))
        output_U = Polygon([o_pt1, pt1, pt2, pt3, pt4, o_pt2, o_pt3, o_pt4])


    if rd_id == 1:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)
        pt1 = Point(line4.interpolate(interp, normalized=True))
        dummy_x = Point(line3.interpolate(offset_x, normalized=True))
        dummy_y = Point(line4.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt2.x, dummy_x.y - o_pt2.y])
        vec_y = np.array([dummy_y.x - o_pt2.x, dummy_y.y - o_pt2.y])
        pt2 = Point((pt1.x + vec_x[0], pt1.y + vec_x[1]))
        pt3 = Point((pt2.x + vec_y[0], pt2.y + vec_y[1]))
        pt4 = Point((pt3.x - vec_x[0], pt3.y - vec_x[1]))
        output_U = Polygon([o_pt1, o_pt2, pt1, pt2, pt3, pt4, o_pt3, o_pt4])


    if rd_id == 2:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)
        pt1 = Point(line6.interpolate(interp, normalized=True))
        dummy_x = Point(line6.interpolate(offset_x, normalized=True))
        dummy_y = Point(line5.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt3.x, dummy_x.y - o_pt3.y])
        vec_y = np.array([dummy_y.x - o_pt3.x, dummy_y.y - o_pt3.y])
        pt2 = Point((pt1.x + vec_y[0], pt1.y + vec_y[1]))
        pt3 = Point((pt2.x + vec_x[0], pt2.y + vec_x[1]))
        pt4 = Point((pt3.x - vec_y[0], pt3.y - vec_y[1]))
        output_U = Polygon([o_pt1, o_pt2, o_pt3, pt1, pt2, pt3, pt4, o_pt4])

    if rd_id == 3:
        interp = (1.0 - offset_x) * np.random.normal(0.5, 0.01, 1)
        pt1 = Point(line8.interpolate(interp, normalized=True))
        dummy_x = Point(line7.interpolate(offset_x, normalized=True))
        dummy_y = Point(line8.interpolate(offset_y, normalized=True))
        vec_x = np.array([dummy_x.x - o_pt4.x, dummy_x.y - o_pt4.y])
        vec_y = np.array([dummy_y.x - o_pt4.x, dummy_y.y - o_pt4.y])
        pt2 = Point((pt1.x + vec_x[0], pt1.y + vec_x[1]))
        pt3 = Point((pt2.x + vec_y[0], pt2.y + vec_y[1]))
        pt4 = Point((pt3.x - vec_x[0], pt3.y - vec_x[1]))
        output_U = Polygon([o_pt1, o_pt2, o_pt3, o_pt4, pt1, pt2, pt3, pt4])

    return output_U



def get_org_layout(geo_list):
    exist, merge, posx, posy, sizex, sizey, shape_pred, b_iou, block, midaxis = geo_list
    exist_mask = exist > 0
    pos = np.stack((posx, posy), axis = 1)
    size = np.stack((sizex, sizey), axis = 1)

    pred_size = size[exist_mask]
    pred_pos = pos[exist_mask]

    pred_size[:, 0] = (pred_size[:, 0] + 0.252) / 2.0
    pred_size[:, 1] = pred_size[:, 1] + 0.479


    block_azimuth, block_bbx = get_block_parameters(block)
    block = norm_block_to_horizonal([block], block_azimuth, block)[0]
    block = block.simplify(2.0)
    asp_rto = get_block_aspect_ratio(block, midaxis)

    pred_pos_sort = np.lexsort((pred_pos[:,1],pred_pos[:,0]))
    pred_pos_xsorted = pred_pos[pred_pos_sort]
    pred_size_xsorted = pred_size[pred_pos_sort]
    pred_pos_xsorted, pred_size_xsorted = modify_pos_size_arr_overlap(pred_pos_xsorted, pred_size_xsorted)
    pred_bldg, pred_pos1, pred_size1 = inverse_warp_bldg_by_midaxis(pred_pos_xsorted, pred_size_xsorted, midaxis, asp_rto)

#############################################################
    shape_pred = shape_pred[pred_pos_sort]
    iou = b_iou[pred_pos_sort]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.axis('off')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)    
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    plt.plot(*block.exterior.xy, color = road_contour_color, linewidth=2)
    plt.plot(*midaxis.coords.xy, color = 'red', linewidth=2)
    if pred_pos1.shape[0] > 0:
        for ii in range(len(pred_bldg)):
            if iou[ii] >=1 or iou[ii]<=0:
                iou[ii] = 0.95
            if pred_bldg[ii].minimum_rotated_rectangle.geom_type != 'Polygon':
                continue
            shp = 0 if shape_pred[ii] == 1 else shape_pred[ii]
            shaped_bldg = get_geometry_by_shape_iou(shp, iou[ii], pred_bldg[ii])
            # ax1.fill(*shaped_bldg.exterior.xy, color = bldg_colors[shp], alpha=0.75)
            ax1.fill(*shaped_bldg.exterior.xy, color = bldg_colors[0], alpha=0.75)
            plt.plot(*shaped_bldg.exterior.xy, color = bldg_contour_color, alpha=0.95, linewidth=0.75)
    # plt.savefig(os.path.join(self.pretrained_dir, str(self.cur_idx) + '.png'), bbox_inches='tight',pad_inches = 0)
    fig1.canvas.draw()
    data = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
    # print(data.shape)
    plt.clf()
    plt.close()
    return data