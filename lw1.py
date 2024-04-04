import cv2
import numpy as np


def cv2_threshold(
    source_image: np.array,
    max_val: int,
    block_size: int,
    C: int,
) -> np.array:
    """Native OpenCV adaptiveThreshold

    Args:
        source_image: Source grayscaled image.
        max_val: Non-zero value assigned to the pixels for which the condition is satisfied.
        block_size: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel.
        C: Constant subtracted from the mean.

    Returns:
        np.array: thresholded image
    """
    thr = cv2.adaptiveThreshold(
        source_image,
        max_val,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )
    return thr


def adaptive_threshold_python(
    source_image: np.array,
    max_val: int,
    block_size: int,
    C: int,
) -> np.array:
    
    """ Adaptive threshold implementation using Pythong + Numpy.
    Uses ADAPTIVE_THRESH_MEAN_C strategy:
        The threshold value T(x, y) is a means of 
            [xi: xi+block_size, yi: yi+block_size] minus C
    
    By default using something like cv2.THRESHOLD_BINARY strategy.
            
    Args:
        source_image: Source grayscaled image.
        max_val: Non-zero value assigned to the pixels for which the condition is satisfied.
        block_size: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel.
        C: Constant subtracted from the mean.
        
    Returns:
        np.array: thresholded image
    """
    
    assert len(source_image.shape) == 2, "Image is not gray_scaled"
    
    n_rows, n_cols = source_image.shape
    res_image = np.zeros_like(source_image)
    for i in range(0, n_rows, block_size):
        for j in range(0, n_cols, block_size):
            # print(f"Taking block : [{i}:{i+block_size}, {j}:{j+block_size}]")
            block = source_image[i:i+block_size, j:j+block_size]
            local_mean = block.mean()
            threshold = local_mean - C
            res_image[i :i+block_size, j: j+block_size] = (
                block > threshold
                ) * max_val
            
    return res_image


def main() -> None:
    
    # reading image
    image = cv2.imread("images/opened-book-gray.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # parameters for adaptive threshold
    max_val = 255
    block_size = 51
    C = 15
    
    # CV2 adaptive threshold
    
    threshold_cv2 = cv2_threshold(
        source_image=image,
        max_val=max_val,
        block_size=block_size,
        C=C,
    )
    
    threshold_p1 = adaptive_threshold_python(
        source_image=image,
        max_val=max_val,
        block_size=block_size,
        C=C,
    )
    
    # For compfort fitting inside window screen
    WINDOW_SIZE = (720, 720)
    resized_image = cv2.resize(image, WINDOW_SIZE)
    resized_thr_cv2 = cv2.resize(threshold_cv2, WINDOW_SIZE)
    resized_thr_p1 = cv2.resize(threshold_p1, WINDOW_SIZE)
    

    # Showing source image
    cv2.putText(
        resized_image,
        "INITIAL Image",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    cv2.imshow("Adaptive Thresholding Comparison", resized_image)
    
    while True:
    # If '1' is pressed, switch to OpenCV thresholded image
        key = cv2.waitKey(0)
        
        if key == ord("1"):
            cv2.putText(
                resized_image,
                "INITIAL Image",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Adaptive Thresholding Comparison", resized_image)
        
        elif key  == ord('2'):
            cv2.putText(
                resized_thr_cv2,
                "OPENCV",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Adaptive Thresholding Comparison", resized_thr_cv2)

        # If '3' is pressed, switch to custom thresholded image
        elif key == ord('3'):
            
            cv2.putText(
                resized_thr_p1,
                "PYTHON+NUMPY",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Adaptive Thresholding Comparison", resized_thr_p1)

        # If 'q' is pressed, exit the loop
        elif key == ord('q'):
            break
        
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
    