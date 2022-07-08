from PIL import Image
import os.path  
import glob

def convertImage(filePath,outdir,width=1024,height=1024):
    img=Image.open(filePath)
    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir,os.path.basename(filePath)))

for imgFile in glob.glob("./*.jpg"):
    convertImage(imgFile,"../image_processed/")