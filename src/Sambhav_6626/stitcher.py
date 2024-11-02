import pdb
import glob
import cv2
import os
import math
import numpy as np


class PanaromaStitcher():
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        images = [cv2.imread(img_path) for img_path in all_images]
        
        keypoints_list, descriptors_list = self.detect_features(images)
        
        # Step 2: Match features sequentially between image pairs
        all_img1_pts, all_img2_pts = self.match_features_sequentially(images, keypoints_list, descriptors_list)
        
        
        # Step 3: Estimate transformation matrices and homographies
        transformation_matrices = self.calculate_transformation_matrices(all_img1_pts, all_img2_pts)
        all_img1_pts_homogeneous = [np.array([np.append(pt, 1) for pt in img1_pts]) for img1_pts in all_img1_pts]
        all_img2_pts_homogeneous = [np.array([np.append(pt, 1) for pt in img2_pts]) for img2_pts in all_img2_pts]
        # print(all_img1_pts_homogeneous)
        H_final_list = self.calculate_final_homographies(images, all_img1_pts_homogeneous, all_img2_pts_homogeneous, transformation_matrices)
        final_Homography = np.array(H_final_list) /  np.array(H_final_list)[:, -1, -1, np.newaxis, np.newaxis]
        num_images = len(images)
        H_chain_final = self.compute_homography_chains(H_final_list, num_images)
        # Step 4: Stitch images to form panorama
        stitched_image = self.stitch_images(images, H_chain_final)
        cropped_image = self.crop_black_border(stitched_image)
        
        return cropped_image, final_Homography
        # Collect all homographies calculated for pair of images and return
        # homography_matrix_list =[]
        # # Return Final panaroma
        # #transformation_matrices = calculate_transformation_matrices(all_img1_pts, all_img2_pts)
        # stitcher = cv2.Stitcher_create()
        # status, stitched_image = stitcher.stitch([cv2.imread(im) for im in all_images])
        # stitched_image = cv2.imread(all_images[0])
        #####
        
        # return stitched_image, homography_matrix_list 
    
    def detect_features(self, images):
        keypoints_list = []
        descriptors_list = []
        
        for img in images:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
        return keypoints_list, descriptors_list
            
    def match_features_sequentially(self, images, keypoints_list, descriptors_list):
        all_img1_pts = []
        all_img2_pts = []

        for i in range(len(images) - 1):
            des1 = descriptors_list[i]
            des2 = descriptors_list[i + 1]
            matches = self.bf.knnMatch(des1, des2, k=2)

            good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

            img1_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in good_matches])
            img2_pts = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in good_matches])

            all_img1_pts.append(img1_pts)
            all_img2_pts.append(img2_pts)
        return all_img1_pts, all_img2_pts
    
    def compute_scale_and_centroid(self,points):
        centroid = np.mean(points, axis=0)
        n = points.shape[0]
        squared_diffs = np.sum((points[:, 0] - centroid[0])**2 + (points[:, 1] - centroid[1])**2)
        scale = np.sqrt(squared_diffs / (2 * n))
        return scale, centroid
    
    def compute_transformation_matrix(self,scale, centroid):
        T = np.array([
            [1/scale, 0, -centroid[0]/scale],
            [0, 1/scale, -centroid[1]/scale],
            [0, 0, 1]
        ])
        return T
    def calculate_transformation_matrices(self, all_img1_pts, all_img2_pts):
        transformation_matrices = []
        for img1_pts, img2_pts in zip(all_img1_pts, all_img2_pts):
            s, m_centroid = self.compute_scale_and_centroid(img1_pts)
            s_prime, m_prime_centroid =self.compute_scale_and_centroid(img2_pts)
            T = self.compute_transformation_matrix(s, m_centroid)
           
            T_prime = self.compute_transformation_matrix( s_prime, m_prime_centroid)
            transformation_matrices.append((T, T_prime))
        return transformation_matrices
    
    def select_random_points(self,pts1, pts2, num_points=4):
        if pts1.shape != pts2.shape or pts1.shape[0] < num_points:
            raise ValueError("Both point sets must have the same shape and at least 'num_points' points.")
        indices = np.random.choice(pts1.shape[0], size=num_points, replace=False)
    
        selected_pts1 = pts1[indices]
        selected_pts2 = pts2[indices]
    
        return selected_pts1, selected_pts2
    
    def apply_transformation(self,points, transformation_matrix):
        transformed_points = np.zeros_like(points)
    
        for i in range(points.shape[0]):
            point = points[i]
            transformed_point = transformation_matrix @ point
    
            if transformed_point[2] != 0:
                transformed_points[i] = transformed_point / transformed_point[2]
            else:
                print('error rectify may need after')
                transformed_points[i] = 0
    
        return transformed_points
    
    def compute_homography_matrix(self,pts1, pts2):
        if pts1.shape != pts2.shape or pts1.shape[0] < 4:
            raise ValueError("There must be at least 4 points in both sets with the same shape.")
    
        # Number of point correspondences
        num_points = pts1.shape[0]
    
        # Create the matrix A (2 * num_points x 9)
        A = []
        for i in range(num_points):
            x1, y1,z1 = pts1[i, 0], pts1[i, 1],pts1[i,2]
            x2, y2,z2 = pts2[i, 0], pts2[i, 1],pts2[i,2]
            A.append([0, 0, 0, -z2*x1, -z2*y1, -z2*z1, y2*x1, y2*y1, y2*z1])
            A.append([z2*x1, z2*y1, z2*z1, 0, 0, 0,  -x2*x1, -x2*y1, -x2*z1])
    
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape((3, 3))
        H = H
        return H
    
    def compute_inliers(self,G, test_1, test_12, threshold=5):
        transformed_final_1 = np.zeros_like(test_1)
        
        for K in range(test_1.shape[0]):
            point = test_1[K]
            transformed_point = G @ point.T
            transformed_final_1[K] = transformed_point
    
            if transformed_final_1[K, 2] != 0:
                transformed_final_1[K] /= transformed_final_1[K, 2]
            else:
                transformed_final_1[K] = np.inf
    
        d_squared = np.sum((test_12 - transformed_final_1)**2, axis=-1)
        d_squared_fkme = np.sqrt(d_squared)
    
        inlier_indices = np.where(d_squared_fkme < threshold)[0]
    
        return inlier_indices, transformed_final_1
    
    def calculate_final_homographies(self, images, all_img1_pts, all_img2_pts, transformation_matrices):
        max_inliers_list = []
        G_best_list = []
        H_final_list = []  # Store the final refined homographies
        w=0.28
        p=0.99
        s=4
        N=np.log10(1-p)/np.log10(1-(w**s))
        N=int(N)
        for idx in range(len(images) - 1):
            max_inliers = 0
            G_best = None
            transformed_test_1 = self.apply_transformation(all_img1_pts[idx], transformation_matrices[idx][0])
            transformed_test_12 = self.apply_transformation(all_img2_pts[idx], transformation_matrices[idx][1])

            for _ in range(N):  # RANSAC iterations
                selected_1, selected_2 = self.select_random_points(transformed_test_1, transformed_test_12)
                H = self.compute_homography_matrix(selected_1, selected_2)
                # print("g")
                trans_12_in = np.linalg.inv(transformation_matrices[idx][1])
                H_original = trans_12_in @ H @ transformation_matrices[idx][0]
       # Compute inliers for the current homography
                inlier_indices, transformed_final_1 = self.compute_inliers(H_original, all_img1_pts[idx], all_img2_pts[idx], threshold=5)
                # Compute inliers for this homography
              
                num_inliers = len(inlier_indices)
                
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    G_best =  H_original

            max_inliers_list.append(max_inliers)
            G_best_list.append(G_best)
            
            # Final homography calculation using best inliers
            inlier_indices, _ = self.compute_inliers(G_best, all_img1_pts[idx], all_img2_pts[idx], threshold=5)
            best_test_1 = all_img1_pts[idx][inlier_indices]
            best_test_12 = all_img2_pts[idx][inlier_indices]
            # print(G_best)
            transformed_test_1 = self.apply_transformation(best_test_1, transformation_matrices[idx][0])
            transformed_test_12 = self.apply_transformation(best_test_12, transformation_matrices[idx][1])
            # print(np.shape(transformed_test_1),np.shape(transformed_test_1))
            # Refine final homography
            H_final = self.compute_homography_matrix(transformed_test_1, transformed_test_12)
            trans_12_in = np.linalg.inv(transformation_matrices[idx][1])
            H_FINAL = trans_12_in @ H_final @ transformation_matrices[idx][0]

            H_final_list.append(H_FINAL)

        return H_final_list
    
    
    def check_if_good_homography(H):
        '''
        source https://answers.opencv.org/question/2588/check-if-homography-is-good/
        '''
        if len(H) == 0:
            return False
    
        det = H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1];
        if (det < 0.0):
            print("1")
            return False
    
        N1 = math.sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0]);
        if (N1 > 4 or N1 < 0.1):
            print("2")
            return False
    
        N2 = math.sqrt(H[0, 1] * H[0, 1] + H[1, 1] * H[1, 1]);
        if (N2 > 4 or N2 < 0.1):
            print("3")
            return False
    
        N3 = math.sqrt(H[2, 0] * H[2, 0] + H[2, 1] * H[2, 1]);
        if (N3 > 0.002):
            print("4")
            return False
    
    
        return True
    def BilinearInterpforPixelValue(self,img, pt):
    #Get coordinates of adjacent 4 points
        pt_imin1_jmin1 = img[(math.floor(pt[1])),(math.floor(pt[0]))]
        pt_iplus1_jmin1 = img[math.floor(pt[1]),math.floor(pt[0]+1)]
        pt_imin1_jplus1 = img[math.floor(pt[1]+1),math.floor(pt[0])]
        pt_iplus1_jplus1 =img[math.floor(pt[1]+1),math.floor(pt[0]+1)]
    
        #Calculate weights of points
        xdiff = pt[0] - math.floor(pt[0])
        ydiff = pt[1] - math.floor(pt[1])
        pt_imin1_jmin1_wt= pow(pow(xdiff,2) + pow(ydiff,2),-0.5)
        pt_iplus1_jmin1_wt = pow(pow(1-xdiff,2) + pow(ydiff,2),-0.5)
        pt_imin1_jplus1_wt = pow(pow(xdiff,2) + pow(1-ydiff,2),-0.5)
        pt_iplus1_jplus1_wt = pow(pow(1-xdiff,2) + pow(1-ydiff,2),-0.5)
    
        #Interpolated point
        result_num = pt_imin1_jmin1 * pt_imin1_jmin1_wt + pt_iplus1_jmin1 * pt_iplus1_jmin1_wt + pt_imin1_jplus1 * pt_imin1_jplus1_wt + pt_iplus1_jplus1 * pt_iplus1_jplus1_wt
        result_denom = pt_imin1_jmin1_wt + pt_iplus1_jmin1_wt + pt_imin1_jplus1_wt + pt_iplus1_jplus1_wt
        result = result_num / result_denom
        return result
    def ImageExtent(self,img,H):
        img_corners = np.zeros((3,4))
        img_corners[:,0] = [0,0,1]
        img_corners[:,1] = [0,img.shape[1],1]
        img_corners[:,2] = [img.shape[0],0,1]
        img_corners[:,3] = [img.shape[0],img.shape[1],1]
    
        img_corners_range = np.matmul(H,img_corners)
    
        for i in range(img_corners_range.shape[1]):
            img_corners_range[:,i] = img_corners_range[:,i]/img_corners_range[-1,i]
    
        return img_corners_range[0:2,:]
    def create_distance_weight_mask(self,shape):
        """Create a weight mask where pixels closer to center have higher weights, with added Gaussian blur."""
        height, width = shape[:2]
        y, x = np.ogrid[:height, :width]
        
        # Calculate distance from each edge
        dist_left = x.reshape(-1, width)
        dist_right = (width - x).reshape(-1, width)
        dist_top = y.reshape(height, -1)
        dist_bottom = (height - y).reshape(height, -1)
        
        # Stack all distances and find minimum
        dist_to_edge = np.minimum(np.minimum(dist_left, dist_right), 
                                  np.minimum(dist_top, dist_bottom))
        
        # Normalize weights to [0, 1]
        weights = dist_to_edge / dist_to_edge.max()
        
        # Apply Gaussian blur to smooth edges (hardcoded parameters)
        weights = cv2.GaussianBlur(weights, (21, 21), 10)
    
        # Ensure same shape as input image
        weights = weights.reshape(height, width)
        
        # Add extra dimension if input image has channels
        if len(shape) == 3:
            weights = weights[..., np.newaxis]
            weights = np.repeat(weights, shape[2], axis=2)
        return weights.astype(np.float32)
    def getPanoramicImage(self,range_img, domain_img, H, offsetXY):
        H_inv = np.linalg.inv(H)
        # Create weight mask for domain image with smoothing
        domain_weights = self.create_distance_weight_mask(domain_img.shape)
        
        for i in range(0, range_img.shape[0]):  # Y-coordinate, row
            for j in range(0, range_img.shape[1]):  # X-coordinate, col
                X_domain = np.array([j + offsetXY[0], i + offsetXY[1], 1])
                X_range = np.array(np.matmul(H_inv, X_domain))
                X_range = X_range / X_range[-1]
    
                if (X_range[0] > 0 and X_range[1] > 0 and X_range[0] < domain_img.shape[1] - 1 and X_range[1] < domain_img.shape[0] - 1):
                    # Get pixel value using bilinear interpolation
                    new_pixel = self.BilinearInterpforPixelValue(domain_img, X_range)
                    
                    # Get weight for this pixel from domain image
                    weight = self.BilinearInterpforPixelValue(domain_weights, X_range)
                    
                    # Adjust weight for smoother blending
                    weight *= 0.5  # Hardcoded blend strength
                    
                    # If there's already a value in the range image, blend based on weights
                    if np.any(range_img[i][j]):
                        existing_weight = 1 - weight  # Weight for existing pixel
                        range_img[i][j] = (range_img[i][j] * existing_weight + new_pixel * weight) / (existing_weight + weight)
                    else:
                        range_img[i][j] = new_pixel
                    
        return range_img

    def compute_homography_chains(self,H_final_list, num_images):
        ref_idx = num_images // 2
        H_chain_final = []
    
        for i in range(num_images):
            if i < ref_idx:
                H_chain = np.eye(3)
                for j in range(i, ref_idx):
                    H_chain = H_final_list[j] @ H_chain
                H_chain_final.append(H_chain)
    
            elif i > ref_idx:
                H_chain = np.eye(3)
                for j in range(i - 1, ref_idx - 1, -1):
                    H_chain = np.linalg.inv(H_final_list[j]) @ H_chain
                H_chain_final.append(H_chain)
    
            else:
                H_chain_final.append(np.eye(3))
    
        return H_chain_final
    def stitch_images(self, images,  H_chain_final):
       # Calculate the extent for each image in the final panorama
        corners_list = [
            self.ImageExtent(images[i], H_chain_final[i] / H_chain_final[i][2, 2])
            for i in range(len(images))
        ]

        # Calculate the minimum and maximum x, y coordinates across all images
        min_xy_coord = np.amin(np.amin(corners_list, axis=2), axis=0)
        max_xy_coord = np.amax(np.amax(corners_list, axis=2), axis=0)
        final_img_dim = max_xy_coord - min_xy_coord
        j = int(final_img_dim[1]*1.5)
        k = int(final_img_dim[0]*1.5)
        pan_img = np.zeros((j, k, 3), dtype=np.uint8)
        # Initialize the panorama canvas with computed dimensions
        # pan_img = np.zeros((int(final_img_dim[0]), int(final_img_dim[1]), 3), dtype=np.uint8)

        # Warp each image and place it on the panoramic canvas
        for i in range(len(images)):
            H_normalized = H_chain_final[i] / H_chain_final[i][2, 2]
            pan_img = self.getPanoramicImage(pan_img, images[i], H_normalized, min_xy_coord)

        return pan_img
    def crop_black_border(self, img):
      # Detect non-black pixels
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
      non_black_pixels = np.where(gray > 0)  # Get indices of non-black pixels

      # Find the bounding box of non-black pixels
      min_y, max_y = np.min(non_black_pixels[0]), np.max(non_black_pixels[0])
      min_x, max_x = np.min(non_black_pixels[1]), np.max(non_black_pixels[1])

      # Crop the image based on the bounding box
      cropped_img = img[min_y:max_y + 1, min_x:max_x + 1]

      return cropped_img
   



    
            
     
    

        
    
    
    
    



    
