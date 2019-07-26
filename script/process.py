import cv2, glob, numpy, os
from tqdm import tqdm
#for f in tqdm(glob.glob("output_combined2/*.jpeg") + glob.glob("output_combined2/*.jpg") + glob.glob("output_combined2/*.tif")
#              + glob.glob("output_combined2/*.png")):
for f in tqdm(glob.glob("Test/*.jpeg") + glob.glob("Test/*.jpg") + glob.glob("Test/*.tif")):
    try:
        if 'gb' in f:
            print(f"skip{f}")
            continue
        a=cv2.imread(f)
        if a.shape[0] > a.shape[1]:
            width = 350
            height = int(350 * a.shape[0] / a.shape[1])
        else:
            height = 350
            width = int(350 * a.shape[1] / a.shape[0])
        a=cv2.resize(a, (width,height))
        # b=cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), 50), -4, 128)
        c=cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), 350/30), -4, 128)
        # cv2.imwrite(f.replace("/","/gb_50_"),b)
        cv2.imwrite(f.replace("/","/gb_12_"),c)
        # os.remove(f)
    except:
        print("failed"+f)
print("finished")
# os.system('sudo shutdown now')
