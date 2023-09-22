
import argparse
import numpy as np
import skimage
from pathlib import Path
import torch 


def read_image(filepath):
    image = skimage.io.imread(filepath)
    return image

def resize_image(image,scaling_factor):
    image = skimage.transform.resize(image, (int(image.shape[0]*scaling_factor), int(image.shape[1]*scaling_factor)), anti_aliasing=True)
    return image


def get_flow(image1, image2):
    flow = np.load('/mnt/2tb-hdd/harshaM/DenseFlow/RAFT-DenseFlow-Modified/DyNeRF/Test/01/140_150/flow_up_12.npy')
    return flow

def get_camera_intrinsic_matrix(csv_filepath):
    camera_intrinsics = np.loadtxt(csv_filepath, delimiter=',')
    camera_intrinsics_matrix = camera_intrinsics[0].reshape(3,3)

    return camera_intrinsics_matrix

def get_camera_extrinsic_matrix(csv_filepath):
    camera_extrinsics = np.loadtxt(csv_filepath, delimiter=',')
    camera_extrinsics_matrix = camera_extrinsics[0].reshape(4,4)
    return camera_extrinsics_matrix

def get_cross_matrix(vector):
    tx = vector[0]
    ty = vector[1]
    tz = vector[2]
    cross_matrix = np.array([[0, -tz, ty], 
                             [tz, 0, -tx], 
                             [-ty, tx, 0]])
    return cross_matrix

def get_fundamental_matrix(camera_extrinsics_v_matrix_homogeneous,camera_extrinsics_u_matrix_homogeneous):
    camera_extrinsics_v_to_u_matrix = np.matmul(camera_extrinsics_u_matrix_homogeneous, np.linalg.inv(camera_extrinsics_v_matrix_homogeneous))
    rotation_matrix = camera_extrinsics_v_to_u_matrix[:,0:3]
    translation_vector = camera_extrinsics_v_to_u_matrix[:,3]
    translation_vector_cross = get_cross_matrix(translation_vector)
    fundamental_matrix = np.matmul(rotation_matrix[:-1,:], translation_vector_cross)

    return fundamental_matrix

def get_essential_matrix(camera_intrinsics_v_matrix,camera_intrinsics_u_matrix, fundamental_matrix):
    essential_matrix = np.matmul(np.linalg.inv(camera_intrinsics_u_matrix).T, np.matmul(fundamental_matrix, np.linalg.inv(camera_intrinsics_v_matrix)))
    return essential_matrix

# def get_epipolar_line(x,y,camera_intrinsics_matrix, fundamental_matrix, flow_12):
#     q_s_v = np.array([x+flow_12[x,y,0], y+flow_12[x,y,1], 1]).reshape(3,1)
#     camera_constant_matrix = np.matmul(camera_intrinsics_matrix,np.matmul(fundamental_matrix,np.linalg.inv(camera_intrinsics_matrix)))
#     epipolar_line = np.matmul(camera_constant_matrix,q_s_v)
#     return epipolar_line

def get_perpendicular_distance(epipolar_line, q_s_u):
    dot_product = torch.sum(torch.mul(epipolar_line, q_s_u),dim=(2))
    norm = torch.linalg.norm(epipolar_line[:,:,:-1], dim=(2))
    perpendicular_distance = torch.div(dot_product, norm)
    return perpendicular_distance




def for_one_image_pair(camera_id,cross_camera_id, image_id_1, image_id_2):
    images_dirpath = Path(f'DyNeRF/{camera_id:02}/images')
    image1_filepath = images_dirpath / f'{image_id_1:04}.jpg'
    image2_filepath = images_dirpath / f'{image_id_2:04}.jpg'
    image1 = read_image(image1_filepath)
    image2 = read_image(image2_filepath)

    image1 = resize_image(image1, 0.5)
    image2 = resize_image(image2, 0.5)

    flow_12 = get_flow(image1, image2)

 

    resized_flow = np.resize(flow_12, (flow_12.shape[0]-2, flow_12.shape[1], flow_12.shape[2]))
    camera_intrinsics_v_matrix = get_camera_intrinsic_matrix(f'DyNeRF/{camera_id:02}/CameraIntrinsics.csv')
    camera_intrinsics_u_matrix = get_camera_intrinsic_matrix(f'DyNeRF/{cross_camera_id:02}/CameraIntrinsics.csv')

    camera_extrinsics_v_matrix = get_camera_extrinsic_matrix(f'DyNeRF/{camera_id:02}/CameraExtrinsics.csv')
    camera_extrinsics_u_matrix = get_camera_extrinsic_matrix(f'DyNeRF/{cross_camera_id:02}/CameraExtrinsics.csv')

    fundamental_matrix = get_fundamental_matrix(camera_extrinsics_v_matrix,camera_extrinsics_u_matrix)
    essential_matrix = get_essential_matrix(camera_intrinsics_v_matrix,camera_intrinsics_u_matrix, fundamental_matrix)
    x1,y1 = np.meshgrid(np.arange(0, image1.shape[1]), np.arange(0, image1.shape[0]))
    I_t_v_grid = np.stack((x1,y1), axis=2)

    I_s_v_grid = I_t_v_grid + resized_flow
    I_s_v_grid_homogeneous = np.concatenate((I_s_v_grid, np.ones((I_s_v_grid.shape[0], I_s_v_grid.shape[1], 1))), axis=2)
    I_s_v_grid_homogeneous = I_s_v_grid_homogeneous.reshape(I_s_v_grid_homogeneous.shape[0], I_s_v_grid_homogeneous.shape[1], 3, 1)
    essential_matrix = essential_matrix.reshape(1,1,3,3)

    epipolar_line = torch.matmul(torch.from_numpy(essential_matrix), torch.from_numpy(I_s_v_grid_homogeneous)).reshape(I_s_v_grid_homogeneous.shape[0], I_s_v_grid_homogeneous.shape[1], 3)
    I_s_u_hat = torch.rand(image1.shape[0], image1.shape[1], 3)
    perpendicular_distance = get_perpendicular_distance(epipolar_line, I_s_u_hat)
    loss = perpendicular_distance
    print(loss.shape)



# def main():
     


    






if __name__ == '__main__':
    # main()
    for_one_image_pair(1,9, 140, 150)



