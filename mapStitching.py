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
                print("수평 스티칭을 수행합니다.")
                result_width = w1 + w2
                result_height = max(h1, h2)
            elif direction == 'vertical':
                print("수직 스티칭을 수행합니다.")
                result_width = max(w1, w2)
                result_height = h1 + h2
            else:
                print("Error: 'horizontal' 또는 'vertical' 방향을 선택해야 합니다.")
                return base_image

            warped_image = cv2.warpPerspective(new_image, M, (result_width, result_height))
            
            warped_image[0:h1, 0:w1] = base_image

            # # 4. 결과 시각화

            # result_img = cv2.drawMatches(img[i], kp1, img[i+1], kp2, matches[:1000], None, 
            #                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # if i != len(images) - 1:
            #     resultImgGray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
            #     gray[i + 1] = resultImgGray
            
        # 5. 결과 보여주기
        cv2.imshow("Matching Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result_img

    def stitch_images_orb_v2(self, base_image, new_image, direction='horizontal', maxFeatures=2000):
        # 1. 특징점 검출 및 매칭 (방향과 무관)
        gray1 = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=maxFeatures)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            print("Warning: 특징점을 찾을 수 없어 스티칭을 건너뜁니다.")
            return base_image # 문제가 생기면 기준 이미지를 그대로 반환

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:200]

        debug_img = cv2.drawMatches(base_image, kp1, new_image, kp2, good_matches, None, flags=2)

        cv2.imshow("Debug Matches", debug_img)
        cv2.imwrite('debug_img.jpg', debug_img)
        cv2.waitKey(0)

        if len(good_matches) < 4:
            print("Warning: 매칭점이 부족하여 스티칭을 건너뜁니다.")
            return base_image

        # 2. 변환 행렬 계산 (방향과 무관)
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 3. 방향에 따라 캔버스 크기 결정
        h1, w1 = base_image.shape[:2]
        h2, w2 = new_image.shape[:2]

        if direction == 'horizontal':
            print("수평 스티칭을 수행합니다.")
            result_width = w1 + w2
            result_height = max(h1, h2)
        elif direction == 'vertical':
            print("수직 스티칭을 수행합니다.")
            result_width = max(w1, w2)
            result_height = h1 + h2
        else:
            print("Error: 'horizontal' 또는 'vertical' 방향을 선택해야 합니다.")
            return base_image

        # 4. 이미지 변환 및 합성
        # 결정된 캔버스 크기로 새 이미지를 변환
        warped_image = cv2.warpPerspective(new_image, M, (result_width, result_height))
        
        # 변환된 이미지 위에 기준 이미지를 덮어쓰기
        warped_image[0:h1, 0:w1] = base_image

        return warped_image

    def phaseCorrelation(self, video_frames: list):
        if not video_frames:
            print("프레임이 없습니다.")
        else:
            # 1. 첫 번째 프레임을 최종 파노라마의 시작으로 설정
            panorama = video_frames[0]
            prev_frame_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)

            # 두 번째 프레임부터 반복
            for i in range(1, len(video_frames)):
                current_frame = video_frames[i]
                current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                # 2. 이전 프레임과 현재 프레임 사이의 Y축 이동량 계산
                # phaseCorrelate는 float32를 입력으로 받으므로 변환
                shift, _ = cv2.phaseCorrelate(np.float32(prev_frame_gray), np.float32(current_frame_gray))
                dx, dy = shift
                
                # 픽셀 단위로 사용하기 위해 정수로 반올림
                # dy는 현재 프레임이 이전 프레임 대비 얼마나 아래로 움직였는가 (카메라는 위로 움직였으므로)
                dy = int(round(dy))

                print(f"Frame {i}와 Frame {i+1} 사이의 이동량: dx={dx}, dy={dy}")

                # dy가 음수이거나 너무 작으면 (겹침이 없거나 움직임이 거의 없으면) 건너뜁니다.
                # 겹치는 부분이 최소한 10픽셀은 되어야 의미 있는 스티칭이 됩니다.
                if dy <= 0 or dy >= current_frame.shape[0]: # dy가 이미지 높이보다 크면 겹침 없음
                    print(f"Skipping frame {i+1} due to insufficient/invalid overlap (dy={dy}).")
                    prev_frame_gray = current_frame_gray # 다음 프레임과의 비교를 위해 업데이트는 함
                    continue
                    
                # 3. 현재 프레임에서 겹치지 않는 아랫부분만 잘라내기
                # current_frame[시작_행:끝_행, 시작_열:끝_열]
                # dy만큼의 윗부분을 제외한 [dy:]부터 아랫부분 전체를 사용 
                new_strip = current_frame[:dy, :]
                # cv2.imshow("New Strip", new_strip)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(f'new_strip_{i}.jpg', new_strip)

                # 4. 기존 파노라마 이미지 아래에 새로운 스트립을 이어 붙이기
                # cv2.vconcat은 이미지의 가로 크기가 동일해야 합니다.
                if panorama.shape[1] != new_strip.shape[1]:
                    # 가로 크기가 다르면 한쪽에 맞춰야 함 (여기서는 panorama에 맞춤)
                    new_strip = cv2.resize(new_strip, (panorama.shape[1], new_strip.shape[0]))

                panorama = cv2.vconcat([new_strip, panorama])

                # 5. 다음 반복을 위해 현재 프레임을 '이전 프레임'으로 업데이트
                prev_frame_gray = current_frame_gray
                
                # 진행 상황 확인 (선택 사항)
                # cv2.imshow("Stitching in progress", cv2.resize(panorama, None, fx=0.2, fy=0.2))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            # cv2.imshow("Final Panorama", panorama)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            new_width = int(panorama.shape[1] * 0.1)
            new_height = int(panorama.shape[0] * 0.1)
            panorama_resized = cv2.resize(panorama, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Final Panorama", panorama_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('stitched_image.jpg', panorama)
            return panorama

def main():
    print(f"현재 스크립트가 사용 중인 OpenCV 버전: {cv2.__version__}")
    image_paths = 'ezgif-split'
    images = load_images(image_paths)
    # stitched_image = stitch_images(images)
    # cv2.imwrite('stitched_image.jpg', stitched_image)

    customStitcher = customStitching()
    # stitched_image = customStitcher.stitch_images_orb_v2(images[0], images[1], 'horizontal', 2000)
    stitched_image = customStitcher.phaseCorrelation(images)
    print("Stitched image saved")

if __name__ == '__main__':
    main()