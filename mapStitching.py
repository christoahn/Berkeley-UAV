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

    # None 값 제거 (이미지 로드 실패한 경우)
    images = [img for img in images if img is not None]

    images = [images[i] for i in range(0, len(images))]
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

    def stitch_images_orb(self, images: list, maxFeatures: int = 1000):
        gray = list()
        orb = cv2.ORB_create(nfeatures=maxFeatures)
        for img in images:
            gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        for i in range(len(images) - 1):
            kp1, des1= (orb.detectAndCompute(gray[i], None))
            kp2, des2 = (orb.detectAndCompute(gray[i+1], None))
            
            # 3. 특징점 매칭
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            matchs = matches[:100]

            # 4. 결과 시각화

            result_img = cv2.drawMatches(img[i], kp1, img[i+1], kp2, matches[:1000], None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            if i != len(images) - 1:
                resultImgGray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
                gray[i + 1] = resultImgGray
            
        # 5. 결과 보여주기
        cv2.imshow("Matching Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result_img


def main():
    print(f"현재 스크립트가 사용 중인 OpenCV 버전: {cv2.__version__}")
    image_paths = 'ezgif-split'
    images = load_images(image_paths)
    # stitched_image = stitch_images(images)
    # cv2.imwrite('stitched_image.jpg', stitched_image)

    customStitcher = customStitching()
    stitched_image = customStitcher.stitch_images_orb(images, 2000)
    cv2.imwrite('stitched_image.jpg', stitched_image)


if __name__ == '__main__':
    main()