import cv2
import glob
import numpy as np
# import matplotlib.pyplot as plt

def load_images(image_paths, image_format='jpg'):
    print(f'Loading images from {image_paths}')
    pattern = f'{image_paths}/*.{image_format}'
    print(f'Searching for: {pattern}')
    image_files = glob.glob(pattern)
    print(f'Found {len(image_files)} image files')
    images = [cv2.imread(path) for path in image_files]

    images = [images[i] for i in range(0, len(images), 10)]
    # images = images[:3]
    print(f'Successfully loaded {len(images)} images')
    return images

def stitch_images(images):
    print(f'Stitching {len(images)} images')
    stitcher = cv2.Stitcher_create()
    (status, stitched_image) = stitcher.stitch(images)
    if status == 0:
        cv2.imshow("stitched_image", stitched_image)
        print('Image stitching completed')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return stitched_image
    else:
        #Error code
        print(f'Stitching status code: {status}')
        if status == cv2.Stitcher_OK:
            print('Stitching successful!')
        elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print('Error: Need more images for stitching')
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print('Error: Homography estimation failed')
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print('Error: Camera parameters adjustment failed')
        elif status == cv2.Stitcher_ERR_PANORAMA_EST_FAIL:
            print('Error: Panorama estimation failed')
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_VERIFY_FAIL:
            print('Error: Camera parameters verification failed')
        else:
            print(f'Unknown error code: {status}')
        print('Error stitching images')
        return None

class customStitching:
    # Stitcher based on SURF
    def __init__(self):
        import cv2
        import numpy as np
        pass
    def stitch_images(self, images: list, threshold: int = 400):
        surf = cv2.xfeatures2d.SURF_create(400)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=2000)
        kp, des = surf.detectAndCompute(gray, None)
        img_result = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)

        return img_result

    def stitch_images_orb(self, images: list, direction = 'vertical', maxFeatures: int = 1000):
        gray = list()
        orb = cv2.ORB_create(nfeatures=maxFeatures)
        for img in images:
            gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        for i in range(len(images) - 1):
            kp1, des1= (orb.detectAndCompute(gray[i], None))
            kp2, des2 = (orb.detectAndCompute(gray[i+1], None))
            
            # Feature Matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:100]

            # Transformation
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # findHomography function will return how should we transformate images for stitching
            # 3 x 3 matrix
            # RANSAC will ignore some outliers
            # M contains the transformation method
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Acutally transformate image
            if direction == 'horizontal':
                print("Horizontal Stitching")
                result_width = w1 + w2
                result_height = max(h1, h2)
            elif direction == 'vertical':
                print("Vertical Stitching")
                result_width = max(w1, w2)
                result_height = h1 + h2
            else:
                print("Error: 'horizontal' 또는 'vertical' 방향을 선택해야 합니다.")
                return base_image

            warped_image = cv2.warpPerspective(new_image, M, (result_width, result_height))
            
            warped_image[0:h1, 0:w1] = base_image
            
        # Result
        cv2.imshow("Matching Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result_img

    def stitch_images_orb_v2(self, base_image, new_image, direction='horizontal', maxFeatures=2000):
        # Feature Detection and Matching
        gray1 = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=maxFeatures)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            print("Warning: Feature points not found, skipping stitching")
            return base_image # If error, return base image

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:200]

        debug_img = cv2.drawMatches(base_image, kp1, new_image, kp2, good_matches, None, flags=2)

        cv2.imshow("Debug Matches", debug_img)
        cv2.imwrite('debug_img.jpg', debug_img)
        cv2.waitKey(0)

        if len(good_matches) < 4:
            print("Warning: Matching points are insufficient, skipping stitching")
            return base_image

        # Transformation Matrix Calculation
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Canvas Size Decision
        h1, w1 = base_image.shape[:2]
        h2, w2 = new_image.shape[:2]

        if direction == 'horizontal':
            print("Horizontal Stitching")
            result_width = w1 + w2
            result_height = max(h1, h2)
        elif direction == 'vertical':
            print("Vertical Stitching")
            result_width = max(w1, w2)
            result_height = h1 + h2
        else:
            print("Error: 'horizontal' or 'vertical' direction must be selected")
            return base_image

        # Image Transformation and Synthesis
        # Transform new image to determined canvas size
        warped_image = cv2.warpPerspective(new_image, M, (result_width, result_height))
        
        # Overwrite base image on transformed image
        warped_image[0:h1, 0:w1] = base_image

        return warped_image

    def phaseCorrelation(self, video_frames: list):
        if not video_frames:
            print("No frames")
        else:
            # Set first frame as starting point of panorama
            panorama = video_frames[0]
            prev_frame_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)

            # Repeat from second frame
            for i in range(1, len(video_frames)):
                current_frame = video_frames[i]
                current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                # Calculate Y-axis movement amount between previous and current frames
                # phaseCorrelate는 float32를 입력으로 받으므로 변환
                shift, _ = cv2.phaseCorrelate(np.float32(prev_frame_gray), np.float32(current_frame_gray))
                dx, dy = shift
                
                # Round to integer for pixel unit
                dx, dy = int(round(dx)), int(round(dy))

                print(f"Frame {i} and Frame {i+1} movement amount: dx={dx}, dy={dy}")

                if dy <= 0 or dy >= current_frame.shape[0]:
                    print(f"Skipping frame {i+1} due to insufficient/invalid overlap (dy={dy}).")
                    prev_frame_gray = current_frame_gray # Update for next frame comparison
                    continue
                    
                # Cut out only the bottom part that does not overlap in the current frame
                new_strip = current_frame[:dy, :]
                # cv2.imshow("New Strip", new_strip)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(f'new_strip_{i}.jpg', new_strip)

                if panorama.shape[1] != new_strip.shape[1]:
                    new_strip = cv2.resize(new_strip, (panorama.shape[1], new_strip.shape[0]))

                panorama = cv2.vconcat([new_strip, panorama])

                prev_frame_gray = current_frame_gray
                
            new_width = int(panorama.shape[1] * 0.1)
            new_height = int(panorama.shape[0] * 0.1)
            panorama_resized = cv2.resize(panorama, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Final Panorama", panorama_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('stitched_image.jpg', panorama)
            return panorama

def main():
    # print(f"Current OpenCV version: {cv2.__version__}")
    image_paths = 'ezgif-split'
    images = load_images(image_paths)

    customStitcher = customStitching()
    # stitched_image = customStitcher.stitch_images_orb_v2(images[0], images[1], 'horizontal', 2000)
    stitched_image = customStitcher.phaseCorrelation(images)
    print("Stitched image saved")

if __name__ == '__main__':
    main()